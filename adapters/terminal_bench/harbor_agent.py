"""psi-agent Harbor adapter for Terminal-Bench evaluation.

Implements Harbor's BaseInstalledAgent interface so that `harbor run` can
install, run, and score psi-agent inside Terminal-Bench task containers.

Usage:
    harbor run -d terminal-bench@3.0 \
        --agent-import-path adapters.terminal_bench.harbor_agent:PsiAgent \
        --model openai/glm-5.3-max \
        --ae PSI_AI_API_KEY=$ZHIPU_API_KEY \
        --ae PSI_AI_BASE_URL=https://open.bigmodel.cn/api/paas/v4

Harbor drives the adapter through three phases per task:
  1. install() — clone psi-agent, install deps, copy workspace
  2. run()     — start AI + Session + CLI inside the container
  3. (harbor runs the verifier and produces result.json)

The adapter delegates container lifecycle, verifier execution, and result
collection entirely to Harbor — no custom manifest or manual docker cp needed.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# In-container paths (must match psi-agent's expectations)
PSI_HOME = "/opt/psi-agent"
WORKSPACE_IN_CONTAINER = f"{PSI_HOME}/workspace"
SOCKS_DIR = "/tmp/psi-socks"
RESULTS_DIR = "/opt/psi-agent/results"


class PsiAgent(BaseInstalledAgent):
    """psi-agent adapter for Harbor Terminal-Bench."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "psi-agent"

    def version(self) -> str | None:
        return os.environ.get("PSI_AGENT_REF", "main")

    # ── Phase 1: Install ───────────────────────────────────────────────

    async def install(self, environment: BaseEnvironment) -> None:
        """Install psi-agent inside the task container."""
        repo = os.environ.get(
            "PSI_AGENT_REPO",
            "https://github.com/genuineknowledge/psi-agent.git",
        )
        ref = os.environ.get("PSI_AGENT_REF", "main")
        workspace_host = os.environ.get(
            "PSI_AGENT_WORKSPACE",
            str(Path.cwd() / "workspaces" / "terminal_bench"),
        )

        # 0. Ensure python3 + pip exist — some task images are non-Python
        #    (e.g. overfull-hbox is a bare LaTeX image), so the uv bootstrap
        #    below would otherwise fail on the very first pip/curl command.
        await self.exec_as_root(
            environment,
            command=(
                "command -v python3 >/dev/null 2>&1 || "
                "(apt-get update -qq >/dev/null 2>&1 && "
                "apt-get install -y -qq python3 python3-pip curl >/dev/null 2>&1) || "
                "(apk add --no-cache python3 py3-pip curl >/dev/null 2>&1) || "
                "true"
            ),
        )

        # 1. Ensure uv is available (can install Python 3.14)
        await self.exec_as_root(
            environment,
            command=(
                "command -v uv >/dev/null 2>&1 || "
                "(pip3 install --break-system-packages --quiet uv 2>/dev/null || "
                " pip install --break-system-packages --quiet uv 2>/dev/null) || "
                "(curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh 2>/dev/null "
                "&& ln -sf $HOME/.local/bin/uv /usr/local/bin/uv)"
            ),
        )

        # 2. Ensure git (TB3 slim containers may lack it), then clone psi-agent
        await self.exec_as_agent(
            environment,
            command=(
                "command -v git >/dev/null 2>&1 || "
                "(apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq git >/dev/null 2>&1) || "
                "(apk add --no-cache git >/dev/null 2>&1) || "
                "true"
            ),
        )
        await self.exec_as_agent(
            environment,
            command=f"rm -rf {PSI_HOME} && git clone --depth 1 --branch {ref} {repo} {PSI_HOME}",
        )

        # 3. Create venv with Python 3.14 and install psi-agent
        await self.exec_as_agent(
            environment,
            command=f"cd {PSI_HOME} && uv venv --python 3.14",
        )
        await self.exec_as_agent(
            environment,
            command=f"cd {PSI_HOME} && UV_NO_DOWNLOAD_INTERPRETER=1 uv pip install -e .",
        )

        # 4. Copy workspace tools into container
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {WORKSPACE_IN_CONTAINER}",
        )
        workspace_host_path = Path(workspace_host)
        if workspace_host_path.exists():
            await environment.upload_dir(
                str(workspace_host_path),
                WORKSPACE_IN_CONTAINER,
            )

        # 5. Write .env using container env vars (set by --agent-env / --ae)
        #    Unquoted heredoc so shell expands ${VAR:-default} inside container
        env_content = (
            "PSI_AI_PROVIDER=${PSI_AI_PROVIDER:-openai}\n"
            "PSI_AI_MODEL=${PSI_AI_MODEL:-glm-5.3-max}\n"
            "PSI_AI_API_KEY=${PSI_AI_API_KEY:-}\n"
            "PSI_AI_BASE_URL=${PSI_AI_BASE_URL:-}\n"
        )
        await self.exec_as_agent(
            environment,
            command=f"cat > {PSI_HOME}/.env << PSI_EOF\n{env_content}PSI_EOF",
        )

        # 6. Prepare socket + result dirs
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {SOCKS_DIR} {RESULTS_DIR}",
        )

    # ── Phase 2: Run ───────────────────────────────────────────────────

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run psi-agent against the task instruction inside the container.

        Starts three processes (AI server → Session → CLI channel) sequentially,
        waits for each socket, then runs the CLI to completion. Harbor's trial
        orchestrator handles the overall timeout.
        """
        ai_sock = f"{SOCKS_DIR}/ai.sock"
        ch_sock = f"{SOCKS_DIR}/ch.sock"

        # Write instruction to a file inside the container (avoids huge argv)
        instr_escaped = instruction.replace("'", "'\\''")
        await self.exec_as_agent(
            environment,
            command=f"cat > {RESULTS_DIR}/instruction.md << 'PSI_EOF'\n{instruction}PSI_EOF",
        )

        # Build a single orchestration script that:
        # 1. Starts AI server in background
        # 2. Waits for ai.sock
        # 3. Starts Session in background
        # 4. Waits for ch.sock
        # 5. Runs CLI in foreground (blocks until agent completes)
        # 6. Cleans up background processes
        script = textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -e
            cd {PSI_HOME}
            export UV_NO_DOWNLOAD_INTERPRETER=1
            source {PSI_HOME}/.env

            # ── Start AI server (background) ──
            rm -f {ai_sock}
            nohup uv run psi-agent ai \\
              --session-socket {ai_sock} \\
              --provider "${{PSI_AI_PROVIDER:-openai}}" \\
              --model "${{PSI_AI_MODEL:-glm-5.3-max}}" \\
              --api-key "${{PSI_AI_API_KEY:-}}" \\
              --base-url "${{PSI_AI_BASE_URL:-}}" \\
              > {RESULTS_DIR}/ai.log 2>&1 &
            AI_PID=$!

            # ── Wait for AI socket ──
            for i in $(seq 1 100); do
              [ -S {ai_sock} ] && break
              sleep 0.3
            done
            if [ ! -S {ai_sock} ]; then
              echo "ERROR: AI server did not start (see ai.log)" >&2
              kill -9 $AI_PID 2>/dev/null || true
              exit 1
            fi

            # ── Start Session (background) ──
            rm -f {ch_sock}
            nohup uv run psi-agent session \\
              --workspace {WORKSPACE_IN_CONTAINER} \\
              --ai-socket {ai_sock} \\
              --channel-socket {ch_sock} \\
              > {RESULTS_DIR}/session.log 2>&1 &
            SESS_PID=$!

            # ── Wait for Session socket ──
            for i in $(seq 1 100); do
              [ -S {ch_sock} ] && break
              sleep 0.3
            done
            if [ ! -S {ch_sock} ]; then
              echo "ERROR: Session did not start (see session.log)" >&2
              kill -9 $AI_PID $SESS_PID 2>/dev/null || true
              exit 1
            fi

            # ── Run CLI (foreground, blocks until agent completes) ──
            MSG="$(cat {RESULTS_DIR}/instruction.md)"
            uv run psi-agent channel cli \\
              --session-socket {ch_sock} \\
              --message "$MSG" \\
              > {RESULTS_DIR}/agent_output.log 2>&1
            CLI_RC=$?

            # ── Cleanup ──
            kill -9 $AI_PID $SESS_PID 2>/dev/null || true
            rm -f {ai_sock} {ch_sock}

            exit $CLI_RC
        """)

        # Write the script into the container and execute it
        script_path = f"{RESULTS_DIR}/run_psi_agent.sh"
        script_escaped = script.replace("'", "'\\''")
        await self.exec_as_agent(
            environment,
            command=f"cat > {script_path} << 'PSI_EOF'\n{script}PSI_EOF",
        )
        try:
            await self.exec_as_agent(
                environment,
                command=f"chmod +x {script_path} && {script_path}",
            )
        finally:
            # 保存容器内 ai.log 到 pilot_results/<case>/，供 generate_report 解析真实 usage
            case = os.environ.get("PSI_AGENT_CASE", "")
            workdir = os.environ.get("TB_BENCH_WORKDIR", "/root/psi-agent-bench-v2")
            try:
                for _log in ("ai.log", "session.log", "agent_output.log"):
                    _out = await self.exec_as_agent(
                        environment, f"cat /opt/psi-agent/results/{_log} 2>/dev/null"
                    )
                    if case and _out:
                        _dir = Path(workdir) / "pilot_results" / case
                        _dir.mkdir(parents=True, exist_ok=True)
                        (_dir / _log).write_text(str(_out))
            except Exception:
                pass

    # ── Helpers ─────────────────────────────────────────────────────────
