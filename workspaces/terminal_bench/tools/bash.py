"""Bash tool — execute shell commands locally inside the task container.

When the agent runs INSIDE the task container (the recommended architecture),
this tool runs commands directly on the local filesystem — no `docker exec`
bridging needed, so cwd / env / pty / signals / processes are fully native.
"""

from __future__ import annotations

import asyncio
import os
import pty
import shlex
import select
import fcntl
import termios
import struct


async def bash(command: str, timeout_seconds: int = 300) -> str:
    """Run a shell command locally (inside the container), cwd = /app.

    Args:
        command: The shell command to run. Executed with `bash -lc` starting
            from /app.
        timeout_seconds: Max seconds to wait for the command to finish.

    Returns:
        Combined stdout/stderr, with a trailing exit code on failure.
    """
    # Ensure we always start from /app, the standard task workdir.
    workdir = os.environ.get("PSI_PILOT_WORKDIR", "/app")

    wrapped = f"cd {shlex.quote(workdir)} && {command}"

    # --- Allocate a PTY so interactive / curses-style programs behave ---
    # This is the single biggest difference vs subprocess.PIPE: programs that
    # call isatty(1) see a real terminal, prompts appear, progress bars work.
    try:
        pid, fd = pty.fork()
    except (OSError, AttributeError):
        # Fallback for systems without pty support (non-Unix, rare)
        proc = await asyncio.create_subprocess_exec(
            "bash", "-lc", wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"[Error] Command timed out after {timeout_seconds}s: {command}"
        out = stdout.decode(errors="replace").rstrip()
        if proc.returncode != 0:
            out += f"\n[Exit code: {proc.returncode}]"
        return out or "(no output)"

    if pid == 0:
        # --- Child: exec the shell ---
        os.chdir(workdir)
        os.environ["TERM"] = os.environ.get("TERM", "xterm-256color")
        os.execvp("bash", ["bash", "-lc", wrapped])
        # unreachable

    # --- Parent: read from PTY with timeout ---
    # Set the PTY to non-blocking so we can poll with a deadline.
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    # Set a sane window size so programs that query TIOCGWINSZ don't complain.
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ,
                    struct.pack("HHHH", 40, 160, 0, 0))
    except Exception:
        pass

    chunks: list[bytes] = []
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    exit_status: int | None = None

    loop = asyncio.get_event_loop()

    def _waitpid_nonblock() -> tuple[int, int] | None:
        try:
            return os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return (pid, 0)

    while True:
        now = loop.time()
        if now >= deadline:
            # Timeout: kill the whole process group with SIGKILL.
            try:
                os.killpg(os.getpgid(pid), 9)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
            # Drain whatever is left in the PTY before bailing.
            await asyncio.sleep(0.2)
            try:
                while True:
                    data = os.read(fd, 65536)
                    if not data:
                        break
                    chunks.append(data)
            except (BlockingIOError, OSError):
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            return (
                f"[Error] Command timed out after {timeout_seconds}s: {command}"
                f"\nOutput so far:\n" + b"".join(chunks).decode(errors="replace").rstrip()
            )

        # Check if the child has exited.
        wp = _waitpid_nonblock()
        if wp and wp[0] != 0:
            _, status = wp
            if os.WIFEXITED(status):
                exit_status = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                exit_status = 128 + os.WTERMSIG(status)
            else:
                exit_status = 1
            # Drain remaining PTY output.
            await asyncio.sleep(0.05)
            try:
                while True:
                    data = os.read(fd, 65536)
                    if not data:
                        break
                    chunks.append(data)
            except (BlockingIOError, OSError):
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            break

        # Wait until the PTY is readable (or a short tick for the deadline).
        try:
            r, _, _ = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: select.select([fd], [], [], 0.1)),
                timeout=max(0.05, deadline - now),
            )
        except asyncio.TimeoutError:
            continue

        if fd in r:
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                continue
            except OSError:
                break
            if not data:
                # EOF — child closed the PTY slave side.
                continue
            chunks.append(data)

    out = b"".join(chunks).decode(errors="replace").rstrip()
    # Strip raw CR that the PTY tty layer injects before every LF.
    out = out.replace("\r\n", "\n").replace("\r", "\n")

    if exit_status not in (0, None):
        out += f"\n[Exit code: {exit_status}]"

    return out or "(no output)"
