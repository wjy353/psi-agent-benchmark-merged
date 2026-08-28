"""Read tool — read a file's contents locally inside the task container.

When the agent runs INSIDE the container, this tool reads directly from the
container filesystem — no `docker exec` indirection, no escaping issues.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


def _resolve(path: str, workdir: str = "/app") -> str:
    # Relative paths are interpreted relative to the workdir (default /app).
    if path.startswith("/"):
        return path
    return os.path.join(workdir, path)


async def read(file_path: str, offset: int = 0, limit: int = 0) -> str:
    """Read file contents from the local (container) filesystem.

    Args:
        file_path: Path to the file (absolute, or relative to /app).
        offset: Line number to start reading from (0 = beginning).
        limit: Max lines to read (0 = no limit).

    Returns:
        File contents, or an error message if unreadable.
    """
    workdir = os.environ.get("PSI_PILOT_WORKDIR", "/app")
    resolved = _resolve(file_path, workdir)

    try:
        # Open synchronously — the OS read syscall is typically fast enough
        # for the file sizes an agent should ever inspect. We still run it
        # inside a thread executor to avoid blocking the event loop.
        def _do_read() -> str:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                if offset <= 0 and limit <= 0:
                    return f.read()
                lines = f.readlines()
                start = max(0, offset)
                if limit > 0:
                    lines = lines[start:start + limit]
                else:
                    lines = lines[start:]
                return "".join(lines)

        content = await asyncio.to_thread(_do_read)
    except FileNotFoundError:
        return f"[Error] File not found: {file_path} (resolved to {resolved})"
    except IsADirectoryError:
        return f"[Error] {file_path} is a directory, not a file"
    except PermissionError:
        return f"[Error] Permission denied: {file_path}"
    except OSError as e:
        return f"[Error] Cannot read {file_path}: {e}"

    content = content.rstrip()
    return content or "(empty file)"
