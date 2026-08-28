"""Edit tool — make a precise string replacement inside the task container.

Content is read/written directly via Python io, so special characters (tabs,
newlines, Unicode, regex metacharacters) are handled literally — no shell
escaping, no cat/heredoc ambiguity, no stdin byte-level framing artifacts.
"""

from __future__ import annotations

import asyncio
import os


def _resolve(path: str, workdir: str = "/app") -> str:
    if path.startswith("/"):
        return path
    return os.path.join(workdir, path)


async def edit(file_path: str, old_string: str, new_string: str) -> str:
    """Replace a unique old_string with new_string in a local file (container).

    Args:
        file_path: Path to the file (absolute, or relative to /app).
        old_string: Exact text to find (must appear exactly once).
        new_string: Text to replace it with.

    Returns:
        Success or error message.
    """
    workdir = os.environ.get("PSI_PILOT_WORKDIR", "/app")
    resolved = _resolve(file_path, workdir)

    def _do_edit() -> str:
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            return f"[Error] File not found: {file_path} (resolved to {resolved})"
        except PermissionError:
            return f"[Error] Permission denied: {file_path}"
        except IsADirectoryError:
            return f"[Error] {file_path} is a directory, not a file"
        except OSError as e:
            return f"[Error] Cannot read {file_path}: {e}"

        count = content.count(old_string)
        if count == 0:
            return f"[Error] old_string not found in {file_path}"
        if count > 1:
            return (
                f"[Error] old_string appears {count} times in {file_path}; "
                "must be unique to edit safely"
            )

        new_content = content.replace(old_string, new_string, 1)

        try:
            # Write atomically via temp + rename so a crash never leaves
            # a half-written file for the verifier to trip over.
            tmp = f"{resolved}.edit.tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)
            os.replace(tmp, resolved)
        except OSError as e:
            return f"[Error] Cannot write edited {file_path}: {e}"

        return f"[OK] Replaced 1 occurrence in {resolved}"

    return await asyncio.to_thread(_do_edit)
