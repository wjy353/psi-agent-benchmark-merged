"""Docker 容器管理 — 启动/停止/操作 Terminal-Bench 任务容器。

NEW ARCHITECTURE (v2): the agent runs INSIDE the task container, not on the
host. We only use `docker run` to start the container with appropriate bind
mounts (psi-agent runtime + workspace + task instruction + result logs), and
then `docker exec` to launch psi-agent ai/session/cli processes *inside* the
container. All tool calls operate on the container's local filesystem with
full tty / env / cwd / signal semantics.

Mount layout inside the container:
  /opt/psi-agent                  -> psi-agent source (uv project) + venv symlinks
  /opt/psi-agent/workspace        -> terminal_bench workspace
  /opt/psi-agent/task             -> task.toml + instruction.md (read-only)
  /opt/psi-agent/results          -> per-case result dir (logs are written here)
  /tmp/psi-socks                  -> ai/session unix sockets
"""

import json
import os
import shlex
import shutil
import subprocess
import time
import tomllib
from pathlib import Path


# ── In-container paths (consumed by the agent + our docker exec wrappers) ───
PSI_IN_CONTAINER       = "/opt/psi-agent"
WORKSPACE_IN_CONTAINER = f"{PSI_IN_CONTAINER}/workspace"
TASK_IN_CONTAINER      = f"{PSI_IN_CONTAINER}/task"
RESULTS_IN_CONTAINER   = f"{PSI_IN_CONTAINER}/results"
SOCKS_IN_CONTAINER     = "/tmp/psi-socks"


def run_cmd(cmd, *, cwd=None, timeout=None, capture=True, log_fn=None):
    """Execute a shell command on the HOST (docker cli, harbor, etc.)."""
    if log_fn:
        log_fn(f"CMD: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=capture, text=True, timeout=timeout
    )
    if log_fn:
        if result.returncode != 0:
            tail = ((result.stdout or "") + "\n" + (result.stderr or "")).strip().splitlines()[-50:]
            for line in tail:
                log_fn(f"  {line}")
        else:
            log_fn("  RC=0 OK")
    return result


def image_exists(tag):
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", tag], capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def ensure_task_downloaded(name, tasks_dir, harbor_bin, log_fn=None):
    """Download a Terminal-Bench task definition if missing.

    Returns the task directory Path or None on failure.
    """
    task_dir = tasks_dir / name
    if task_dir.exists() and (task_dir / "task.toml").exists():
        return task_dir

    if log_fn:
        log_fn(f"Downloading task {name}...")
    result = run_cmd(
        [
            harbor_bin, "task", "download",
            f"terminal-bench/{name}",
            "--export", "--output-dir", str(tasks_dir), "--overwrite",
        ],
        timeout=300, capture=True, log_fn=log_fn,
    )
    return task_dir if result.returncode == 0 else None


def parse_task_toml(task_dir):
    """Parse task.toml into a dict."""
    with open(task_dir / "task.toml", "rb") as f:
        return tomllib.load(f)


def start_env_container(container_name, image_tag, *,
                        psi_dir: Path, workspace_dir: Path,
                        task_dir: Path, result_dir: Path,
                        log_fn=None):
    """Start the task environment container and mount the agent runtime.

    Bind mounts (host -> container):
      psi_dir                    -> /opt/psi-agent
      workspace_dir              -> /opt/psi-agent/workspace (OVERRIDE the copy
                                    from psi_dir so the latest local version wins)
      task_dir                   -> /opt/psi-agent/task  (ro, instruction.md etc.)
      result_dir                 -> /opt/psi-agent/results (rw, ai.log/session.log)
      $HOME/.local/share/uv/python  -> same path inside container  (CRITICAL so
                                    the uv-created venv's python symlink resolves)
      /logs/verifier             -> anonymous volume for the verifier output

    The container stays alive via `sleep infinity`; the agent + verifier are
    launched later via `docker exec`.
    """
    # ── Pre-cleanup on HOST: stale PSI sockets from previous (v1) runs were
    #    left on /tmp and accidentally picked up by psi-agent's default
    #    fallback socket name (psi-{ai,ch}-{container_name}.sock on host /tmp).
    #    We nuke them BEFORE starting the case so old listeners can't interfere.
    import glob as _glob
    for stale in _glob.glob(f"/tmp/psi-ai-{container_name}.sock") + _glob.glob(f"/tmp/psi-ch-{container_name}.sock"):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass

    run_cmd(["docker", "rm", "-f", container_name], capture=False, log_fn=log_fn)

    psi_dir        = Path(psi_dir).resolve()
    workspace_dir  = Path(workspace_dir).resolve()
    task_dir       = Path(task_dir).resolve()
    result_dir     = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── Mount UV python install dir into container (same absolute path) ──
    # psi-agent 2026.08 requires-python>=3.14. On the host, `uv run` downloads
    # CPython 3.14 to $HOME/.local/share/uv/python/cpython-3.14-* and creates a
    # venv symlink .venv/bin/python -> that path. We bind-mount the parent
    # directory read-only into the container so the symlink resolves WITHOUT
    # triggering a fresh 35MB cp3.14 download + 6 minute wait per case.
    host_home = Path(os.path.expanduser("~"))
    uv_py_root = host_home / ".local" / "share" / "uv" / "python"
    extra_mounts = []
    if uv_py_root.exists():
        candidates = sorted(uv_py_root.glob("cpython-3.14*"))
        if candidates:
            extra_mounts += [
                "-v", f"{candidates[0]}:/root/.local/share/uv/python/{candidates[0].name}:ro",
            ]
            if log_fn:
                log_fn(f"Mounting uv-python {candidates[0].name} into container")

    # ── Also bind-mount the host's `uv` CLI binary directly into the
    #    container at the canonical location. The 2.x TB images ship with
    #    `uv` preinstalled, but the 3.0+ "slim" task images (e.g.
    #    bun-sourcemap-leak) do NOT — the psi-agent scripts always exec
    #    `uv run psi-agent ...` which fails with "uv: not found" on 3.0.
    #    Mounting the single ~59MB static binary read-only is faster and
    #    more predictable than trying to pip/apt install it per case.
    host_uv_bin = Path(shutil.which("uv") or "/usr/local/bin/uv")
    if host_uv_bin.is_file():
        extra_mounts += [
            "-v", f"{host_uv_bin}:/usr/local/bin/uv:ro",
        ]
        if log_fn:
            log_fn(f"Mounting host uv binary {host_uv_bin} into container as /usr/local/bin/uv")

    cmd = [
        "docker", "run", "-d", "--name", container_name,
        # ── share the host network so the AI API is reachable ──
        "--network", "host",
        # ── standard TB verifier output volume ──
        "-v", "/logs/verifier",
        # ── psi-agent runtime (source + uv venv) ──
        "-v", f"{psi_dir}:{PSI_IN_CONTAINER}:rw",
        # ── terminal_bench workspace (local version takes precedence) ──
        "-v", f"{workspace_dir}:{WORKSPACE_IN_CONTAINER}:ro",
        # ── task instruction + task.toml ──
        "-v", f"{task_dir}:{TASK_IN_CONTAINER}:ro",
        # ── result/logs dir (ai.log, session.log, agent_output.log) ──
        "-v", f"{result_dir}:{RESULTS_IN_CONTAINER}:rw",
        # ── shared memory / IPC (some TB task images rely on /dev/shm) ──
        "--ipc", "host",
        # ── uv python + uv binary mounts (explained above) ──
        *extra_mounts,
        image_tag, "sleep", "infinity",
    ]
    result = run_cmd(cmd, timeout=120, capture=True, log_fn=log_fn)
    if result.returncode != 0:
        return False

    # ── Bootstrapping: create standard dirs, fix symlinks, probe python ──
    # NOTE: bootstrap is a bash script with `${...}` expansions. We write it as
    # a plain Python f-string using ONLY the constants at the top of this
    # file (SOCKS_IN_CONTAINER, RESULTS_IN_CONTAINER) — which never contain
    # braces — and double-brace everything bash-related (${VAR} -> ${{VAR}})
    # so Python's f-string parser doesn't trip over them.
    bootstrap = (
        "set -e\n"
        f"mkdir -p /logs/verifier /app {SOCKS_IN_CONTAINER} {RESULTS_IN_CONTAINER} /root/.local/share/uv/python\n"
        "\n"
        "# ---------- Critical: expose uv-downloaded CPython as system python3 ----------\n"
        "# The host psi-agent venv was built by `uv run` which:\n"
        "#   (a) downloads CPython to $HOME/.local/share/uv/python/cpython-*\n"
        "#   (b) creates .venv/bin/python -> /root/.local/share/uv/python/cpython-*/bin/python3.X\n"
        "# We bind-mounted that exact tree. Now make /usr/local/bin/python3 / python /\n"
        "# python3.X all point to that interpreter so uv never thinks the venv is broken\n"
        "# and never auto-downloads a 35MB 6-minute interpreter per case.\n"
        "UV_PY_DIR=\"$(ls -d /root/.local/share/uv/python/cpython-3.1[0-9]*/bin 2>/dev/null | head -n1 || true)\"\n"
        "if [ -n \"$UV_PY_DIR\" ] && [ -x \"$UV_PY_DIR/python3\" ]; then\n"
        "  ln -sf \"$UV_PY_DIR/python3\"    /usr/local/bin/python3\n"
        "  ln -sf \"$UV_PY_DIR/python3\"    /usr/local/bin/python\n"
        "  for alt in \"$UV_PY_DIR\"/python3.*; do\n"
        "    [ -x \"$alt\" ] || continue\n"
        "    short=\"${alt##*/}\"\n"
        "    target=\"/usr/local/bin/$short\"\n"
        "    [ -e \"$target\" ] || ln -sf \"$alt\" \"$target\"\n"
        "  done\n"
        "  export PATH=\"/usr/local/bin:$UV_PY_DIR:$PATH\"\n"
        "elif command -v python3 >/dev/null 2>&1; then\n"
        "  :\n"
        "elif command -v python >/dev/null 2>&1; then\n"
        "  ln -sf \"$(command -v python)\" /usr/local/bin/python3 || true\n"
        "fi\n"
        "\n"
        "# ---------- uv binary ----------\n"
        "if command -v uv >/dev/null 2>&1 && ! command -v /usr/local/bin/uv >/dev/null 2>&1; then\n"
        "  ln -sf \"$(command -v uv)\" /usr/local/bin/uv || true\n"
        "fi\n"
        "\n"
        "# ---------- Essential infrastructure bootstrap ----------\n"
        "# Many TB task images (especially 3.0 'slim' variants) lack basic tools\n"
        "# that agents need: ca-certificates (HTTPS fails without it, even for\n"
        "# `uv pip install`), curl (downloading files), unzip (extracting\n"
        "# archives). We auto-install them if missing. This is idempotent and\n"
        "# adds ~5-15 seconds on first run per container, zero if already present.\n"
        "MISSING_PKGS=\"\"\n"
        "# Detect package manager\n"
        "if command -v apt-get >/dev/null 2>&1; then\n"
        "  PKG_MGR=apt\n"
        "elif command -v dnf >/dev/null 2>&1; then\n"
        "  PKG_MGR=dnf\n"
        "elif command -v apk >/dev/null 2>&1; then\n"
        "  PKG_MGR=apk\n"
        "else\n"
        "  PKG_MGR=unknown\n"
        "fi\n"
        "# ca-certificates: CRITICAL — without it, HTTPS requests fail (including\n"
        "# `uv pip install` from PyPI). Even if the base image ships some CA\n"
        "# certs, an explicit install ensures they are up-to-date.\n"
        "if [ ! -d /etc/ssl/certs ] || [ \"$(find /etc/ssl/certs -name '*.pem' 2>/dev/null | wc -l)\" -lt 5 ]; then\n"
        "  case $PKG_MGR in\n"
        "    apt)  MISSING_PKGS=\"$MISSING_PKGS ca-certificates\" ;;\n"
        "    dnf)  MISSING_PKGS=\"$MISSING_PKGS ca-certificates\" ;;\n"
        "    apk)  MISSING_PKGS=\"$MISSING_PKGS ca-certificates\" ;;\n"
        "  esac\n"
        "fi\n"
        "# curl: HIGH — agents frequently need to download files from the internet.\n"
        "# wget is an acceptable alternative; only install curl if neither exists.\n"
        "if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then\n"
        "  case $PKG_MGR in\n"
        "    apt)  MISSING_PKGS=\"$MISSING_PKGS curl\" ;;\n"
        "    dnf)  MISSING_PKGS=\"$MISSING_PKGS curl\" ;;\n"
        "    apk)  MISSING_PKGS=\"$MISSING_PKGS curl\" ;;\n"
        "  esac\n"
        "fi\n"
        "# unzip: MEDIUM — agents may need to extract .zip archives.\n"
        "if ! command -v unzip >/dev/null 2>&1; then\n"
        "  case $PKG_MGR in\n"
        "    apt)  MISSING_PKGS=\"$MISSING_PKGS unzip\" ;;\n"
        "    dnf)  MISSING_PKGS=\"$MISSING_PKGS unzip\" ;;\n"
        "    apk)  MISSING_PKGS=\"$MISSING_PKGS unzip\" ;;\n"
        "  esac\n"
        "fi\n"
        "# Install missing packages (best-effort, don't fail the whole bootstrap)\n"
        "if [ -n \"$MISSING_PKGS\" ]; then\n"
        "  echo \"[bootstrap] Installing missing packages:$MISSING_PKGS\"\n"
        "  case $PKG_MGR in\n"
        "    apt)\n"
        "      apt-get update -qq 2>/dev/null && \\\n"
        "      apt-get install -y --no-install-recommends $MISSING_PKGS 2>/dev/null && \\\n"
        "      rm -rf /var/lib/apt/lists/* 2>/dev/null || \\\n"
        "      echo \"[bootstrap] WARN: apt install failed (non-fatal)\"\n"
        "      ;;\n"
        "    dnf)\n"
        "      dnf install -y $MISSING_PKGS 2>/dev/null || \\\n"
        "      echo \"[bootstrap] WARN: dnf install failed (non-fatal)\"\n"
        "      ;;\n"
        "    apk)\n"
        "      apk add --no-cache $MISSING_PKGS 2>/dev/null || \\\n"
        "      echo \"[bootstrap] WARN: apk add failed (non-fatal)\"\n"
        "      ;;\n"
        "    *)\n"
        "      echo \"[bootstrap] WARN: unknown package manager, cannot install $MISSING_PKGS\"\n"
        "      ;;\n"
        "  esac\n"
        "fi\n"
        "\n"
        "# Show environment header for the log\n"
        "echo \"[bootstrap] uname:    $(uname -a)\"\n"
        "echo \"[bootstrap] python3:  $(command -v python3 || echo MISSING)\"\n"
        "echo \"[bootstrap] $(python3 --version 2>&1 || echo python3 MISSING)\"\n"
        "echo \"[bootstrap] bash:     $(command -v bash    || echo MISSING)\"\n"
        "echo \"[bootstrap] uv:       $(command -v uv      || echo MISSING)\"\n"
        "echo \"[bootstrap] curl:     $(command -v curl    || echo MISSING)\"\n"
        "echo \"[bootstrap] unzip:    $(command -v unzip   || echo MISSING)\"\n"
        "echo \"[bootstrap] ca-certs: $( [ -d /etc/ssl/certs ] && echo 'present' || echo MISSING)\"\n"
        "ls -la /app 2>/dev/null || echo \"[bootstrap] /app is empty or missing\"\n"
    )
    subprocess.run(
        ["docker", "exec", container_name, "bash", "-lc", bootstrap],
        capture_output=True, text=True, timeout=120,
    )
    return True


def build_verifier_image(task_dir, tag, log_fn=None):
    """Build a separate verifier image if required."""
    dockerfile = task_dir / "tests" / "Dockerfile"
    if not dockerfile.exists():
        return False
    if image_exists(tag):
        if log_fn:
            log_fn(f"Verifier image {tag} exists, skip build")
        return True
    result = run_cmd(
        [
            "docker", "build", "-t", tag,
            "-f", str(dockerfile), str(task_dir / "tests"),
        ],
        timeout=3600, capture=True, log_fn=log_fn,
    )
    return result.returncode == 0


def _in_container_env_extras() -> list[str]:
    """Return a bash snippet that exports all PSI_* / TB_* env vars from
    the host into the container so the AI client inherits credentials.
    """
    passthrough_prefixes = ("PSI_", "TB_", "OPENAI_", "DEEPSEEK_", "ANTHROPIC_",
                            "API_KEY", "BASE_URL", "MODEL", "PROVIDER", "HTTP_",
                            "HTTPS_", "NO_PROXY", "UV_")
    lines = []
    for key, val in os.environ.items():
        if any(key.startswith(p) for p in passthrough_prefixes) or key in {"HOME", "PATH"}:
            safe_val = shlex.quote(val)
            lines.append(f"export {key}={safe_val}")
    lines.append(f"export PSI_PILOT_WORKDIR=/app")
    # Prevent uv from trying to auto-download a CPython interpreter when the
    # bind-mounted venv symlink should already be correct. Falls back per
    # bootstrap if interpreter not found (no download, just error).
    lines.append("export UV_NO_DOWNLOAD_INTERPRETER=1")
    return lines


def run_agent(container_name, task_dir, result_dir, psi_dir, uv_bin, workspace, agent_timeout, log_fn=None):
    """Run psi-agent FULLY INSIDE the task container via docker exec.

    Architecture inside the container:
      - `psi-agent ai`        listens on {SOCKS_IN_CONTAINER}/ai.sock
      - `psi-agent session`   connects to ai.sock, uses workspace in-container
      - `psi-agent channel cli` sends instruction.md content to session

    All three processes are `docker exec` children; their stdout/stderr is
    redirected into result_dir (which is a bind-mount back to the host) so
    the host can follow progress without additional copying.

    Returns (returncode, status) where status ∈ {"finished", "timeout"}.
    """
    task_dir    = Path(task_dir).resolve()
    result_dir  = Path(result_dir).resolve()
    psi_dir     = Path(psi_dir).resolve()

    # Sockets live inside the container on tmpfs so they disappear with it.
    ai_sock = f"{SOCKS_IN_CONTAINER}/ai.sock"
    ch_sock = f"{SOCKS_IN_CONTAINER}/ch.sock"

    # ── Passthrough host env vars into the container ─────────────────────
    env_bootstrap = "\n".join(_in_container_env_extras())

    # ── Ensure uv is available inside the container ──────────────────────
    # `uv` itself is almost always available on the host (system-wide at
    # /usr/local/bin/uv or in PATH). If we got here through setup.sh, the
    # bootstrap phase has already symlinked it into /usr/local/bin. We only
    # fall back to pip install if uv is truly missing from the image.
    # CRITICAL: set UV_NO_DOWNLOAD_INTERPRETER=1 so uv never thinks the
    # psi-agent venv is broken and starts a 35MB 6-minute cpython download.
    ensure_uv = f"""
export UV_NO_DOWNLOAD_INTERPRETER=1
# Prefer the uv binary the user configured (TB_UV_BIND env / absolute path on
# the host which is a valid path inside the container due to matching layout).
if command -v {uv_bin} >/dev/null 2>&1; then
  echo "[agent-env] uv at {uv_bin}: $({uv_bin} --version 2>&1)"
  exit 0
fi
if command -v uv >/dev/null 2>&1; then
  ln -sf "$(command -v uv)" /usr/local/bin/{os.path.basename(uv_bin)} 2>/dev/null || true
  echo "[agent-env] uv: $(uv --version 2>&1) (system uv, symlinked to {os.path.basename(uv_bin)})"
  exit 0
fi
pip install --quiet uv 2>/dev/null || pip3 install --quiet uv 2>/dev/null || true
if command -v uv >/dev/null 2>&1; then
  ln -sf "$(command -v uv)" /usr/local/bin/{os.path.basename(uv_bin)} 2>/dev/null || true
  echo "[agent-env] uv installed via pip: $(uv --version 2>&1)"
  exit 0
fi
echo "[agent-env] WARN: uv not available — agent commands will likely fail"
"""

    subprocess.run(
        ["docker", "exec", container_name, "bash", "-lc",
         f"{env_bootstrap}\n{ensure_uv}"],
        capture_output=True, text=True, timeout=120,
    )

    # ── Build the three psi-agent sub-command scripts ────────────────────
    # We write each invocation as a small shell script in RESULTS_IN_CONTAINER
    # and then `docker exec` that script. This keeps quoting sane and lets
    # the host see exactly what command was executed (scripts are preserved
    # in the result bind-mount).

    # ── Purge stale logs + scripts from a previous run. The scripts use
    #    append redirection (>>) which can interleave old and new content,
    #    giving a false impression that uv is still downloading interpreters
    #    or that the wrong tool paths are being used.
    for stale in ["ai.log", "session.log", "agent_output.log",
                  "run_ai.sh", "run_session.sh", "run_cli.sh",
                  "instruction.md"]:
        p = result_dir / stale
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    instruction_md = (task_dir / "instruction.md").read_text(encoding="utf-8")
    # Write instruction.md into the result dir so it's captured for debugging,
    # then the cli process reads it from there (avoids huge argv).
    instr_in_container = f"{RESULTS_IN_CONTAINER}/instruction.md"
    (result_dir / "instruction.md").write_text(instruction_md, encoding="utf-8")

    ai_script = f"""#!/usr/bin/env bash
set -e
{env_bootstrap}
cd {PSI_IN_CONTAINER}
rm -f {ai_sock}
exec {uv_bin} run psi-agent ai \\
  --session-socket {ai_sock} \\
  --provider "${{PSI_AI_PROVIDER:-}}" \\
  --model "${{PSI_AI_MODEL:-}}" \\
  --api-key "${{PSI_AI_API_KEY:-}}" \\
  --base-url "${{PSI_AI_BASE_URL:-}}" \\
  >>{RESULTS_IN_CONTAINER}/ai.log 2>&1
"""
    sess_script = f"""#!/usr/bin/env bash
set -e
{env_bootstrap}
cd {PSI_IN_CONTAINER}
rm -f {ch_sock}
# Give the AI server a moment to create its socket.
for i in $(seq 1 50); do
  [ -S {ai_sock} ] && break
  sleep 0.2
done
exec {uv_bin} run psi-agent session \\
  --workspace "{WORKSPACE_IN_CONTAINER}" \\
  --ai-socket {ai_sock} \\
  --channel-socket {ch_sock} \\
  >>{RESULTS_IN_CONTAINER}/session.log 2>&1
"""
    cli_script = f"""#!/usr/bin/env bash
set -u
{env_bootstrap}
cd {PSI_IN_CONTAINER}
# Give the session a moment to create its socket.
for i in $(seq 1 100); do
  [ -S {ch_sock} ] && break
  sleep 0.2
done
MSG="$(cat {instr_in_container})"
exec {uv_bin} run psi-agent channel cli \\
  --session-socket {ch_sock} \\
  --message "$MSG" \\
  >>{RESULTS_IN_CONTAINER}/agent_output.log 2>&1
"""

    for name, script in (("run_ai.sh", ai_script),
                         ("run_session.sh", sess_script),
                         ("run_cli.sh", cli_script)):
        (result_dir / name).write_text(script, encoding="utf-8", newline="\n")

    # ── Launch ai + session as backgrounded docker exec tasks ───────────
    # We use detached docker exec (-d) so these processes survive the host
    # subprocess.Popen lifecycle. Their PIDs live in the container's pid
    # namespace, which we reference by container+script name for cleanup.
    def _detached_exec(script_name: str) -> str:
        """Start a script in-container in detached mode; return the exec ID."""
        inner_path = f"{RESULTS_IN_CONTAINER}/{script_name}"
        res = subprocess.run(
            ["docker", "exec", "-d", container_name,
             "bash", "-lc", f"chmod +x {inner_path} && {inner_path}"],
            capture_output=True, text=True, timeout=15,
        )
        return res.stdout.strip() if res.returncode == 0 else ""

    if log_fn:
        log_fn(f"Starting agent inside container {container_name} (timeout {agent_timeout}s)")

    ai_exec_id    = _detached_exec("run_ai.sh")
    sess_exec_id  = _detached_exec("run_session.sh")

    # ── Wait a bit for ai + session to finish binding sockets ───────────
    time.sleep(6)

    # ── Run the CLI (foreground docker exec) with overall agent timeout ──
    cli_inner = f"{RESULTS_IN_CONTAINER}/run_cli.sh"
    try:
        # subprocess.run for docker exec cli; the outer timeout corresponds
        # to the case agent_timeout budget.
        cli_res = subprocess.run(
            ["docker", "exec", container_name,
             "bash", "-lc", f"chmod +x {cli_inner} && {cli_inner}"],
            capture_output=False, timeout=agent_timeout,
        )
        return cli_res.returncode, "finished"
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    finally:
        if log_fn:
            log_fn("Stopping in-container agent processes (ai + session)")
        # Kill anything still running — we match by the container's exec IDs
        # and additionally nuke any uv/python process owned by root that has
        # psi-agent on its cmdline. This is overkill but safe.
        kill_script = f"""
set +e
# Stop any exec-IDs we still have handles for
if command -v docker >/dev/null; then
  # we are inside docker exec so docker cli is not present — use pkill instead
  :
fi
pkill -9 -f 'psi-agent ai'      2>/dev/null || true
pkill -9 -f 'psi-agent session' 2>/dev/null || true
pkill -9 -f 'psi-agent channel' 2>/dev/null || true
pkill -9 -f 'uv run psi-agent'  2>/dev/null || true
# Clean up sockets so the next case doesn't inherit stale files
rm -f {ai_sock} {ch_sock}
echo "[agent-cleanup] done"
"""
        subprocess.run(
            ["docker", "exec", container_name, "bash", "-lc", kill_script],
            capture_output=True, text=True, timeout=15,
        )


def run_verifier(container_name, task_dir, task_toml, verifier_tag, result_dir, log_fn=None):
    """Run the verifier and extract reward.

    For ``environment_mode = "separate"`` (TB 3.0), we must copy the agent's
    ``/app`` directory out of the environment container and bind-mount it into
    the verifier container.  ``--volumes-from`` only inherits *named volumes*
    (like ``/logs/verifier``), NOT the container's writable filesystem layer
    where ``/app`` artifacts live.  Without this fix, the verifier sees an
    empty ``/app`` and every test fails regardless of the agent's work.

    For ``environment_mode = "same"`` (TB 2.1), the verifier runs inside the
    same container via ``docker exec``, so ``/app`` is already visible.
    """
    import tempfile as _tf

    verifier = task_toml.get("verifier", {})
    mode = verifier.get("environment_mode", "same")
    if log_fn:
        log_fn(f"Verifier mode: {mode}")

    app_snapshot = None  # will hold the temp-dir Path if we docker-cp /app

    try:
        if mode == "separate":
            verifier_container = f"{container_name}-verifier"
            run_cmd(["docker", "rm", "-f", verifier_container], capture=False, log_fn=log_fn)

            # ── CRITICAL FIX: --volumes-from only inherits named volumes, NOT ──
            #    the container filesystem layer.  /app artifacts live in the
            #    container layer, so we must docker-cp them out and bind-mount
            #    into the verifier container.
            app_snapshot = Path(_tf.mkdtemp(prefix="tb-verifier-app-"))
            if log_fn:
                log_fn(f"Snapshotting /app from {container_name} to {app_snapshot}")
            cp_result = subprocess.run(
                ["docker", "cp", f"{container_name}:/app/", str(app_snapshot)],
                capture_output=True, text=True, timeout=120,
            )
            if cp_result.returncode != 0 and log_fn:
                log_fn(f"  docker cp /app failed: {cp_result.stderr.strip()}")
            else:
                snap_count = len(list(app_snapshot.rglob("*")))
                if log_fn:
                    log_fn(f"  /app snapshot: {snap_count} files")

            run_cmd(
                [
                    "docker", "run", "--rm", "--name", verifier_container,
                    "--volumes-from", container_name,
                    "-v", f"{app_snapshot}/app:/app",
                    verifier_tag, "bash", "/tests/test.sh",
                ],
                timeout=600, capture=False, log_fn=log_fn,
            )
        else:
            subprocess.run(
                ["docker", "cp", str(task_dir / "tests"), f"{container_name}:/tests"],
                capture_output=True,
            )
            run_cmd(
                ["docker", "exec", container_name, "bash", "/tests/test.sh"],
                timeout=600, capture=False, log_fn=log_fn,
            )

        # ── Extract reward ──────────────────────────────────────────────────
        reward = "unknown"
        reward_result = subprocess.run(
            ["docker", "exec", container_name, "cat", "/logs/verifier/reward.txt"],
            capture_output=True, text=True,
        )
        if reward_result.returncode == 0 and reward_result.stdout.strip():
            reward = reward_result.stdout.strip()

        if not reward or reward == "unknown":
            json_result = subprocess.run(
                ["docker", "exec", container_name, "cat", "/logs/verifier/reward.json"],
                capture_output=True, text=True,
            )
            if json_result.returncode == 0:
                try:
                    reward_data = json.loads(json_result.stdout.strip())
                    if isinstance(reward_data, dict) and "reward" in reward_data:
                        reward = str(reward_data["reward"])
                except Exception:
                    pass

        verifier_log = Path(result_dir) / "verifier.log"
        verifier_log.write_text(
            f"verifier RC: {reward_result.returncode}\nreward: {reward}\n",
            encoding="utf-8",
        )
        return reward

    finally:
        # ── Clean up the temp /app snapshot directory ────────────────────────
        if app_snapshot and app_snapshot.exists():
            shutil.rmtree(app_snapshot, ignore_errors=True)
            if log_fn:
                log_fn(f"Cleaned up /app snapshot {app_snapshot}")


def cleanup_container(container_name, log_fn=None):
    """Force-remove the container (kills every in-container process)."""
    run_cmd(["docker", "rm", "-f", container_name], capture=False, log_fn=log_fn)
