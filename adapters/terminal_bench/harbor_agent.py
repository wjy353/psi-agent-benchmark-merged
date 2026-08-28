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
import shutil
import textwrap
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.utils.templating import with_prompt_template

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

        # 1. Ensure uv is available (can install Python 3.14)
        await self.exec_as_root(
            environment,
            command=(
                "pip install --quiet uv 2>/dev/null "
                "|| pip3 install --quiet uv 2>/dev/null "
                "|| (curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null "
                "&& ln -sf $HOME/.local/bin/uv /usr/local/bin/uv)"
            ),
        )

        # 2. Clone psi-agent (shallow, specific ref)
        await self.exec_as_agent(
            environment,
            command=f"rm -rf {PSI_HOME} && git clone --depth 1 --branch {ref} {repo} {PSI_HOME}",
        )

        # 3. Install psi-agent (uv handles Python 3.14)
        await self.exec_as_agent(
            environment,
            command=(
                f"cd {PSI_HOME} && "
                "UV_NO_DOWNLOAD_INTERPRETER=1 "
                f"uv pip install -e . --python 3.14"
            ),
        )

        # 4. Copy workspace tools into container
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {WORKSPACE_IN_CONTAINER}",
        )
        workspace_host_path = Path(workspace_host)
        if workspace_host_path.exists():
            tar_path = "/tmp/psi-workspace.tar"
            shutil.make_archive(
                "/tmp/psi-workspace",
                "tar",
                root_dir=str(workspace_host_path),
            )
            # docker cp via exec_as_root (harbor environment may have copy method)
            await self._copy_into_container(
                environment,
                host_path=tar_path,
                container_path=tar_path,
            )
            await self.exec_as_agent(
                environment,
                command=(
                    f"tar xf {tar_path} -C {WORKSPACE_IN_CONTAINER} "
                    f"&& rm {tar_path}"
                ),
            )
            os.unlink(tar_path) if os.path.exists(tar_path) else None

        # 5. Write .env with model credentials
        env_lines = [
            f"PSI_AI_PROVIDER={os.environ.get('PSI_AI_PROVIDER', 'openai')}",
            f"PSI_AI_MODEL={os.environ.get('PSI_AI_MODEL', 'glm-5.3-max')}",
            f"PSI_AI_API_KEY={os.environ.get('PSI_AI_API_KEY', '')}",
            f"PSI_AI_BASE_URL={os.environ.get('PSI_AI_BASE_URL', '')}",
        ]
        env_content = "\n".join(env_lines) + "\n"
        await self.exec_as_agent(
            environment,
            command=f"cat > {PSI_HOME}/.env << 'PSI_EOF'\n{env_content}PSI_EOF",
        )

        # 6. Prepare socket + result dirs
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {SOCKS_DIR} {RESULTS_DIR}",
        )

    # ── Phase 2: Run ───────────────────────────────────────────────────

    @with_prompt_template
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
        await self.exec_as_agent(
            environment,
            command=f"chmod +x {script_path} && {script_path}",
        )

    # ── Phase 3: Post-run (optional, parse tokens/cost) ────────────────

    async def post_run(
        self,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Parse agent logs for token usage and cost, report to Harbor."""
        log_path = f"{RESULTS_DIR}/agent_output.log"
        result = await self.exec_as_agent(
            environment,
            command=f"cat {log_path} 2>/dev/null || echo ''",
        )
        stdout = result.stdout if hasattr(result, "stdout") else str(result)

        # Parse token usage (psi-agent logs total tokens at the end)
        token_match = re.search(
            r"total_tokens[\":\s]+(\d+)", stdout
        )
        if token_match:
            context.report_tokens(total=int(token_match.group(1)))

        # Parse cost (if logged)
        cost_match = re.search(r"cost[\":\s\$]+([0-9.]+)", stdout)
        if cost_match:
            context.report_cost(usd=float(cost_match.group(1)))

    # ── Helpers ─────────────────────────────────────────────────────────

    async def _copy_into_container(
        self,
        environment: BaseEnvironment,
        host_path: str,
        container_path: str,
    ) -> None:
        """Copy a file from the host into the container.

        Tries environment.copy_to_container first, falls back to docker cp.
        """
        copy_fn = getattr(environment, "copy_to_container", None)
        if callable(copy_fn):
            await copy_fn(host_path, container_path)
        else:
            container_id = getattr(environment, "container_id", "")
            if not container_id:
                container_id = getattr(environment, "container_name", "")
            if container_id:
                import subprocess

                subprocess.run(
                    ["docker", "cp", host_path, f"{container_id}:{container_path}"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
