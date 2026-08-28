"""Write tool — create or overwrite a file locally inside the task container.

When the agent runs INSIDE the container, the write lands directly on the
container filesystem — which is exactly what the verifier will inspect.
No shell escaping is ever needed because content is written via the
Python io module directly, not through a shell heredoc.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


def _resolve(path: str, workdir: str = "/app") -> str:
    if path.startswith("/"):
        return path
    return os.path.join(workdir, path)


async def write(file_path: str, content: str) -> str:
    """Create or overwrite a file locally (inside the container).

    Args:
        file_path: Path to the file (absolute, or relative to /app).
        content: The exact content to write (utf-8).

    Returns:
        Success or error message.
    """
    workdir = os.environ.get("PSI_PILOT_WORKDIR", "/app")
    resolved = _resolve(file_path, workdir)

    def _do_write() -> int:
        parent = os.path.dirname(resolved)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return len(content.encode("utf-8"))

    try:
        size = await asyncio.to_thread(_do_write)
    except PermissionError:
        return f"[Error] Permission denied writing {file_path} (resolved to {resolved})"
    except OSError as e:
        return f"[Error] Cannot write {file_path}: {e}"

    return f"[OK] Written {size} bytes to {resolved}"
