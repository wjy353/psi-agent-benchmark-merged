from __future__ import annotations

import json
import os
import re
import shutil
import socket
import string
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAIA_CONFIG = BENCHMARK_ROOT / "config" / "gaia_subsets.json"
DEFAULT_GAIA_DATA_ROOT = BENCHMARK_ROOT / "data" / "GAIA"
DEFAULT_GAIA_REPORT_ROOT = BENCHMARK_ROOT / "reports" / "gaia-psi-agent"
DEFAULT_GAIA_REPO_ID = "ycyc666/GAIA-bucket"


@dataclass(frozen=True)
class GaiaCase:
    case_id: str
    question: str
    answer: str
    level: str
    metadata: dict[str, Any]
    attachments: list[Path]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md_escape(value: object) -> str:
    return ("" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")


def shorten(value: object, limit: int = 160) -> str:
    text = " ".join(("" if value is None else str(value)).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def fmt_float(value: object, default: str = "-") -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else default


def fmt_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0 / 0 = 0.0%"
    return f"{numerator} / {denominator} = {numerator / denominator:.1%}"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_port(url: str, process: subprocess.Popen, timeout: float) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        raise ValueError(f"URL has no port: {url}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("psi-agent process exited before it was ready")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"psi-agent service did not listen within {timeout}s")


def read_api_key(path: Path) -> str:
    if os.environ.get("PSI_AI_API_KEY"):
        return os.environ["PSI_AI_API_KEY"]
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    text = path.read_text(encoding="utf-8").strip()
    if "=" in text:
        return text.split("=", 1)[1].strip().strip('"').strip("'")
    return text


def cache_env(cache_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PSI_GAIA_CACHE_ROOT"] = str(cache_root)
    env["PSI_APPDATA"] = str(cache_root / "psi-appdata")
    env["UV_CACHE_DIR"] = str(cache_root / "uv")
    env["PIP_CACHE_DIR"] = str(cache_root / "pip")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env["PYTHONPYCACHEPREFIX"] = str(cache_root / "pycache")
    env["HF_HOME"] = str(cache_root / "huggingface")
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.1.0"
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PSI_AGENT"] = "0.1.0"
    clear_dead_local_proxy(env)
    return env


def clear_dead_local_proxy(env: dict[str, str]) -> None:
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        value = env.get(name, "")
        if value in {"http://127.0.0.1:9", "https://127.0.0.1:9"}:
            env.pop(name, None)


def list_subsets(config_path: Path) -> list[tuple[str, dict[str, Any]]]:
    config = load_json(config_path)
    subsets = config.get("subsets")
    if not isinstance(subsets, dict):
        raise ValueError("No subsets object found")
    return [(str(name), subset) for name, subset in subsets.items() if isinstance(subset, dict)]


def load_subset(config_path: Path, subset_name: str) -> tuple[str, dict[str, Any]]:
    config = load_json(config_path)
    if not subset_name:
        subset_name = str(config.get("default_subset") or "")
    subsets = config.get("subsets")
    if not isinstance(subsets, dict) or subset_name not in subsets:
        raise KeyError(f"Subset {subset_name!r} not found in {config_path}")
    subset = subsets[subset_name]
    if not isinstance(subset, dict):
        raise ValueError(f"Subset {subset_name!r} is not an object")
    return subset_name, subset


def prepare_gaia_data(data_root: Path, repo_id: str, revision: str) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    bucket_source = repo_id
    if repo_id.startswith("https://huggingface.co/buckets/"):
        bucket_source = repo_id.removeprefix("https://huggingface.co/buckets/")
    if bucket_source.startswith("hf://"):
        bucket_uri = bucket_source
    else:
        bucket_uri = f"hf://buckets/{bucket_source}"
    hf_exe = shutil.which("hf")
    if hf_exe is not None:
        command = [hf_exe, "buckets", "sync", bucket_uri, str(data_root)]
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            command.extend(["--token", token])
        process = subprocess.run(command, check=False)
        if process.returncode == 0:
            return data_root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Could not download GAIA with `hf buckets sync`, and huggingface_hub is not "
            "installed for dataset-repo fallback. If this is a private bucket, set HF_TOKEN "
            f"and run: hf buckets sync {bucket_uri} {data_root}"
        ) from exc
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(data_root),
            revision=revision,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download GAIA data. For the provided bucket source, make sure you "
            f"have access and run `hf auth login` or set HF_TOKEN, then retry: "
            f"hf buckets sync {bucket_uri} {data_root}"
        ) from exc
    return data_root


def _record_case_id(record: dict[str, Any]) -> str:
    for key in ("task_id", "id", "taskId", "Task ID"):
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError(f"GAIA record has no task id: {record.keys()}")


def _record_question(record: dict[str, Any]) -> str:
    for key in ("Question", "question", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"GAIA record has no question: {record.keys()}")


def _record_answer(record: dict[str, Any]) -> str:
    for key in ("Final answer", "final_answer", "answer", "target"):
        value = record.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _record_level(record: dict[str, Any]) -> str:
    for key in ("Level", "level"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _find_attachments(
    data_root: Path,
    split: str,
    case_id: str,
    record: dict[str, Any],
) -> list[Path]:
    for key in ("file_path", "file_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            candidates = [
                data_root / value,
                data_root / split / Path(value).name,
                data_root / "2023" / split / Path(value).name,
            ]
            found = [path for path in candidates if path.is_file()]
            if found:
                return found
    preferred = data_root / split
    legacy_preferred = data_root / "2023" / split
    if preferred.exists():
        search_roots = [preferred]
    elif legacy_preferred.exists():
        search_roots = [legacy_preferred]
    else:
        search_roots = [data_root]
    ignored_suffixes = {".json", ".jsonl", ".parquet", ".arrow", ".csv"}
    matches: list[Path] = []
    for root in search_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in ignored_suffixes:
                continue
            if case_id in path.name:
                matches.append(path)
    return sorted(matches)


def _case_from_record(data_root: Path, split: str, record: dict[str, Any]) -> GaiaCase:
    case_id = _record_case_id(record)
    return GaiaCase(
        case_id=case_id,
        question=_record_question(record),
        answer=_record_answer(record),
        level=_record_level(record),
        metadata={k: v for k, v in record.items() if k not in {"Question", "Final answer"}},
        attachments=_find_attachments(data_root, split, case_id, record),
    )


def _load_with_datasets(data_root: Path, hf_subset: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        dataset = load_dataset(str(data_root), name=hf_subset, split=split)
    except Exception:
        return []
    return [dict(row) for row in dataset]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "validation", "test"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(item)
    return records


def _load_parquet_records(path: Path) -> list[dict[str, Any]]:
    import_error: ImportError | None = None
    try:
        import pandas as pd
    except ImportError as exc:
        import_error = exc
        pd = None
    if pd is not None:
        frame = pd.read_parquet(path)
        return [dict(row) for row in frame.to_dict(orient="records")]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading GAIA parquet metadata requires pandas with a parquet engine or "
            "pyarrow. Install benchmark dependencies with: "
            "python -m pip install -e ."
        ) from (import_error or exc)
    table = pq.read_table(path)
    return [dict(row) for row in table.to_pylist()]


def _candidate_data_files(data_root: Path, hf_subset: str, split: str) -> list[Path]:
    suffixes = ("*.jsonl", "*.json", "*.parquet")
    if (data_root / "2023" / split).exists():
        search_root = data_root / "2023" / split
    elif (data_root / split).exists():
        search_root = data_root / split
    else:
        search_root = data_root
    files: list[Path] = []
    for pattern in suffixes:
        files.extend(search_root.rglob(pattern))
    split_l = split.lower()
    subset_l = hf_subset.lower()
    preferred_names = {
        "2023_all": "metadata.parquet",
        "2023_level1": "metadata.level1.parquet",
        "2023_level2": "metadata.level2.parquet",
        "2023_level3": "metadata.level3.parquet",
    }
    preferred_name = preferred_names.get(subset_l)

    def score(path: Path) -> tuple[int, str]:
        text = path.as_posix().lower()
        value = 0
        if preferred_name and path.name.lower() == preferred_name:
            value -= 20
        if split_l in text:
            value -= 10
        if subset_l in text:
            value -= 8
        if "metadata" in text:
            value -= 4
        return value, text

    return sorted(files, key=score)


def _load_records_from_files(data_root: Path, hf_subset: str, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in _candidate_data_files(data_root, hf_subset, split):
        try:
            if path.suffix.lower() == ".jsonl":
                batch = _load_jsonl_records(path)
            elif path.suffix.lower() == ".json":
                batch = _load_json_records(path)
            elif path.suffix.lower() == ".parquet":
                batch = _load_parquet_records(path)
            else:
                batch = []
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        usable = [
            record
            for record in batch
            if isinstance(record, dict)
            and any(key in record for key in ("Question", "question", "input"))
            and any(key in record for key in ("task_id", "id", "Task ID"))
        ]
        if usable:
            records.extend(usable)
            break
    if errors:
        marker = data_root / ".gaia_load_errors.txt"
        marker.write_text("\n".join(errors), encoding="utf-8")
    return records


def load_cases(
    data_root: Path,
    hf_subset: str,
    split: str,
    limit: int,
    instance_ids: list[str],
) -> list[GaiaCase]:
    if not data_root.exists():
        raise FileNotFoundError(
            f"GAIA data root does not exist: {data_root}. "
            "Run the download-gaia command first or pass --gaia-data-root."
        )
    records = _load_with_datasets(data_root, hf_subset, split)
    if not records:
        records = _load_records_from_files(data_root, hf_subset, split)
    if not records:
        candidates = [str(path) for path in _candidate_data_files(data_root, hf_subset, split)[:10]]
        error_path = data_root / ".gaia_load_errors.txt"
        extra = ""
        if error_path.exists():
            extra = f" Read errors were written to {error_path}."
        raise RuntimeError(
            f"Could not load GAIA records from {data_root}. "
            f"Expected metadata records for hf_subset={hf_subset!r}, split={split!r}. "
            f"Candidate files: {candidates}.{extra}"
        )
    selected = [_case_from_record(data_root, split, record) for record in records]
    if instance_ids:
        wanted = set(instance_ids)
        selected = [case for case in selected if case.case_id in wanted]
    if limit > 0:
        selected = selected[:limit]
    return selected


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def split_string(s: str, char_list: list[str] | None = None) -> list[str]:
    chars = char_list or [",", ";"]
    pattern = f"[{''.join(chars)}]"
    return re.split(pattern, s)


def question_scorer(model_answer: str, ground_truth: str) -> tuple[bool, str]:
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if not ground_truth:
        return False, "No ground-truth answer is available for this split."
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth), f"Evaluated {model_answer} as a number."
    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False, "Evaluated as a list; list lengths differ."
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems, strict=True):
            if is_float(gt_elem):
                comparisons.append(normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons), f"Evaluated {model_answer} as a comma-separated list."
    return normalize_str(model_answer) == normalize_str(ground_truth), (
        f"Evaluated {model_answer} as a string."
    )


def extract_final_answer(raw_prediction: str) -> str:
    text = raw_prediction.strip()
    if not text:
        return ""
    marker_matches = re.findall(
        r"(?im)(?:final\s+answer|answer)\s*[:：]\s*(.+?)(?:\n|$)",
        text,
    )
    if marker_matches:
        return _clean_extracted_answer(marker_matches[-1])
    if len(text) <= 200:
        return _clean_extracted_answer(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        tail = _clean_extracted_answer(lines[-1])
        if len(tail) <= 80:
            return tail
    return text


def _clean_extracted_answer(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^`+|`+$", "", cleaned).strip()
    cleaned = re.sub(r"^\*\*|\*\*$", "", cleaned).strip()
    cleaned = cleaned.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?[.。]?", cleaned):
        cleaned = cleaned.rstrip(".。")
    return cleaned


def build_prompt(case: GaiaCase, workspace: Path) -> str:
    attachment_lines = []
    for attachment in sorted((workspace / "attachments").glob("*")):
        attachment_lines.append(f"- {attachment.name}: {attachment}")
    attachments = "\n".join(attachment_lines) if attachment_lines else "No attachment files."
    return "\n".join(
        [
            "Please answer the GAIA question below.",
            "",
            "Answer format:",
            "- Return only the final answer.",
            "- Use a number, a short phrase, or a comma-separated list.",
            "- Do not include explanation, markdown, citations, or extra text.",
            "",
            "Available local files:",
            attachments,
            "",
            "Question:",
            case.question,
        ]
    )


def copy_attachments(case: GaiaCase, workspace: Path) -> list[str]:
    attachments_dir = workspace / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in case.attachments:
        dst = attachments_dir / path.name
        shutil.copyfile(path, dst)
        copied.append(str(dst))
    return copied


def start_psi_ai(
    *,
    uv: str,
    psi_root: Path,
    ai_url: str,
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            uv,
            "run",
            "--project",
            str(psi_root),
            "psi-agent",
            "ai",
            "--session-socket",
            ai_url,
        ],
        cwd=str(psi_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return process, log_file


def start_psi_session(
    *,
    uv: str,
    psi_root: Path,
    workspace: Path,
    agent_root: Path,
    appdata: Path,
    ai_url: str,
    session_url: str,
    session_id: str,
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            uv,
            "run",
            "--project",
            str(psi_root),
            "psi-agent",
            "session",
            "--workspace",
            str(workspace),
            "--agent",
            str(agent_root),
            "--appdata",
            str(appdata),
            "--session-id",
            session_id,
            "--ai-socket",
            ai_url,
            "--channel-socket",
            session_url,
        ],
        cwd=str(psi_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return process, log_file


def post_to_session(session_url: str, prompt: str, timeout: float) -> tuple[str, list[dict[str, Any]]]:
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}], "stream": True}).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        f"{session_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    events: list[dict[str, Any]] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            events.append(event)
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
    return "".join(chunks).strip(), events


def stop_process(process: subprocess.Popen | None, log_file: Any | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if log_file is not None:
        log_file.close()


def run_case(
    *,
    case: GaiaCase,
    args: Any,
    psi_root: Path,
    agent_root: Path,
    cache_root: Path,
    report_dir: Path,
    ai_url: str,
    env: dict[str, str],
) -> dict[str, Any]:
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in case.case_id)
    case_dir = report_dir / "cases" / safe_id
    workspace = case_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    copied_attachments = copy_attachments(case, workspace)
    prompt = build_prompt(case, workspace)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "question": case.question,
                "answer": case.answer,
                "level": case.level,
                "metadata": case.metadata,
                "attachments": [str(path) for path in case.attachments],
                "workspace_attachments": copied_attachments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (case_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    session_id = f"gaia-{safe_id}-{uuid.uuid4().hex[:8]}"
    session_url = f"http://127.0.0.1:{free_port()}"
    session_process = None
    session_log = None
    started = time.monotonic()
    error = ""
    raw_prediction = ""
    events: list[dict[str, Any]] = []
    try:
        session_process, session_log = start_psi_session(
            uv=args.uv,
            psi_root=psi_root,
            workspace=workspace,
            agent_root=agent_root,
            appdata=cache_root / "psi-appdata",
            ai_url=ai_url,
            session_url=session_url,
            session_id=session_id,
            env=env,
            log_path=case_dir / "session.log",
        )
        wait_for_port(session_url, session_process, timeout=float(args.start_timeout))
        raw_prediction, events = post_to_session(session_url, prompt, timeout=float(args.timeout))
    except Exception as exc:
        error = repr(exc)
    finally:
        stop_process(session_process, session_log)

    duration = time.monotonic() - started
    prediction = extract_final_answer(raw_prediction)
    passed, explanation = question_scorer(prediction, case.answer)
    if error:
        passed = False
        explanation = f"Infrastructure/runtime error: {error}"
    result = {
        "benchmark": "gaia",
        "case_id": case.case_id,
        "level": case.level,
        "question": case.question,
        "target": case.answer,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "scorer_explanation": explanation,
        "error": error,
        "duration": duration,
        "attachments": copied_attachments,
        "session_id": session_id,
        "session_url": session_url,
        "events": events,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (case_dir / "events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not passed:
        write_case_snapshot(case_dir / "snapshot.md", result)
    return result


def write_case_snapshot(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# Failed GAIA Case Snapshot: {result['case_id']}",
        "",
        f"- Level: {result.get('level')}",
        f"- Score: {fmt_float(result.get('score'))}",
        f"- Duration: {fmt_float(result.get('duration'))} s",
        f"- Error: {result.get('error') or '-'}",
        "",
        "## Question",
        str(result.get("question") or ""),
        "",
        "## Prediction",
        str(result.get("prediction") or ""),
        "",
        "## Raw Prediction",
        str(result.get("raw_prediction") or result.get("prediction") or ""),
        "",
        "## Target",
        str(result.get("target") or ""),
        "",
        "## Scorer Explanation",
        str(result.get("scorer_explanation") or ""),
        "",
        "## Attachments",
    ]
    attachments = result.get("attachments")
    if isinstance(attachments, list) and attachments:
        lines.extend(f"- `{path}`" for path in attachments)
    else:
        lines.append("- No attachment files.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(run_id: str, report_dir: Path) -> Path:
    result_files = sorted((report_dir / "cases").glob("*/result.json"))
    if not result_files:
        raise FileNotFoundError(f"No GAIA case results found in {report_dir}")
    rows = []
    for path in result_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        target = str(row.get("target") or "")
        raw_prediction = str(row.get("raw_prediction") or row.get("prediction") or "")
        extracted = extract_final_answer(raw_prediction)
        if raw_prediction and extracted and extracted != row.get("prediction"):
            row["raw_prediction"] = raw_prediction
            row["prediction"] = extracted
            passed, explanation = question_scorer(extracted, target)
            if not row.get("error"):
                row["passed"] = passed
                row["score"] = 1.0 if passed else 0.0
                row["scorer_explanation"] = explanation
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot_path = path.parent / "snapshot.md"
        if row.get("passed") is True:
            if snapshot_path.exists():
                snapshot_path.unlink()
        elif not snapshot_path.exists():
            write_case_snapshot(snapshot_path, row)
        rows.append(row)
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    infra = sum(1 for row in rows if row.get("error"))
    score_values = [row.get("score") for row in rows if isinstance(row.get("score"), int | float)]
    average_score = sum(score_values) / len(score_values) if score_values else 0.0
    by_level: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for row in rows:
        level = str(row.get("level") or "unknown")
        by_level[level]["total"] += 1
        if row.get("passed") is True:
            by_level[level]["passed"] += 1
    summary = {
        "benchmark": "gaia",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": passed / total if total else 0,
        "average_score": average_score,
        "infra_errors": infra,
        "by_level": dict(by_level),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# GAIA case 子集评测报告",
        "",
        f"- 运行 ID: `{run_id}`",
        f"- 生成时间: {summary['generated_at']}",
        "- 被测 Agent: `psi_agent`",
        "",
        "## 一、执行摘要",
        "",
        f"- 总 case 数: {total}",
        f"- 通过率: {fmt_rate(passed, total)}",
        f"- 平均 score: {fmt_float(average_score)}",
        f"- Infra error: {infra}",
        "",
        "## 二、按 Level 统计",
        "",
        "| Level | 题数 | 通过 | 通过率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for level, item in sorted(by_level.items()):
        lines.append(
            f"| {md_escape(level)} | {item['total']} | {item['passed']} | "
            f"{item['passed'] / item['total']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 三、详细结果表",
            "",
            "| Case ID | Level | Score | 结果 | Prediction | Target | Snapshot |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        case_id = str(row.get("case_id") or "")
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in case_id)
        snapshot = Path("cases") / safe_id / "snapshot.md"
        snapshot_link = (
            f"[查看]({snapshot.as_posix()})"
            if row.get("passed") is not True and (report_dir / snapshot).exists()
            else "-"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(case_id),
                    md_escape(row.get("level")),
                    fmt_float(row.get("score")),
                    "PASS" if row.get("passed") is True else "FAIL",
                    md_escape(shorten(row.get("prediction"))),
                    md_escape(row.get("target")),
                    snapshot_link,
                ]
            )
            + " |"
        )
    lines.extend(["", "## 四、失败 Case 快照", ""])
    failed = [row for row in rows if row.get("passed") is not True]
    if failed:
        for row in failed:
            case_id = str(row.get("case_id") or "")
            safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in case_id)
            rel = (Path("cases") / safe_id / "snapshot.md").as_posix()
            lines.append(f"- `{md_escape(case_id)}`: [{rel}]({rel})")
    else:
        lines.append("- 本次子集没有失败 case。")
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def command_download(args: Any) -> int:
    data_root = Path(args.gaia_data_root).resolve()
    prepare_gaia_data(data_root, args.gaia_repo_id, args.gaia_revision)
    print(f"GAIA data downloaded to: {data_root}")
    return 0


def command_list_subsets(args: Any) -> int:
    for name, subset in list_subsets(Path(args.gaia_subset_file)):
        description = subset.get("description") or ""
        hf_subset = subset.get("hf_subset") or ""
        split = subset.get("split") or ""
        limit = subset.get("limit")
        limit_text = "all" if limit is None else str(limit)
        print(f"{name}: {hf_subset}/{split}, limit={limit_text}")
        if description:
            print(f"  {description}")
    return 0


def command_report(args: Any) -> int:
    report_root = Path(args.report_root).resolve()
    if report_root == (BENCHMARK_ROOT / "reports" / "tau2-psi-agent").resolve():
        report_root = DEFAULT_GAIA_REPORT_ROOT.resolve()
    report_path = write_report(args.run_id, report_root / args.run_id)
    print(report_path)
    return 0


def command_run(args: Any) -> int:
    psi_root = Path(args.psi_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    report_root = Path(args.report_root).resolve()
    if report_root == (BENCHMARK_ROOT / "reports" / "tau2-psi-agent").resolve():
        report_root = DEFAULT_GAIA_REPORT_ROOT.resolve()
    data_root = Path(args.gaia_data_root).resolve()
    subset_name, subset = load_subset(Path(args.gaia_subset_file), args.subset)
    hf_subset = str(args.gaia_hf_subset or subset.get("hf_subset") or "2023_level1")
    split = str(args.gaia_split or subset.get("split") or "validation")
    limit = int(args.limit or subset.get("limit") or 0)
    instance_ids = [str(item) for item in args.instance_ids]
    cases = load_cases(data_root, hf_subset, split, limit, instance_ids)
    run_id = args.run_id or f"gaia-psi-{subset_name}-{datetime.now():%Y%m%d-%H%M%S}"
    report_dir = report_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark: gaia")
    print(f"Subset: {subset_name} ({len(cases)} cases)")
    print(f"HF subset: {hf_subset}; split: {split}")
    for case in cases:
        print(f"- {case.case_id}: level={case.level}, attachments={len(case.attachments)}")
    if args.dry_run:
        print("\nDry run only; no API calls were made.")
        return 0

    cache_root.mkdir(parents=True, exist_ok=True)
    env = cache_env(cache_root)
    env["PSI_AGENT_REPO_ROOT"] = str(psi_root)
    env["PSI_AI_PROVIDER"] = args.provider
    env["PSI_AI_MODEL"] = args.model
    env["PSI_AI_BASE_URL"] = args.base_url
    env["PSI_AI_API_KEY"] = env.get("PSI_AI_API_KEY") or read_api_key(Path(args.api_key_file))
    env.setdefault("DEEPSEEK_API_KEY", env["PSI_AI_API_KEY"])

    default_agent = psi_root / "examples" / "haitun-workspace"
    fallback_agent = BENCHMARK_ROOT / "adapters" / "gaia" / "workspace"
    agent_root = Path(args.gaia_agent_root).resolve() if args.gaia_agent_root else default_agent
    if not agent_root.exists():
        agent_root = fallback_agent

    ai_url = f"http://127.0.0.1:{free_port()}"
    state = {
        "benchmark": "gaia",
        "run_id": run_id,
        "subset": subset_name,
        "hf_subset": hf_subset,
        "split": split,
        "psi_root": str(psi_root),
        "agent_root": str(agent_root),
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "ai_url": ai_url,
        "model": args.model,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cases": [{"case_id": case.case_id, "level": case.level} for case in cases],
    }
    state_path = report_dir / "run_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    ai_process, ai_log = start_psi_ai(
        uv=args.uv,
        psi_root=psi_root,
        ai_url=ai_url,
        env=env,
        log_path=report_dir / "logs" / "psi-ai.log",
    )
    exit_code = 0
    try:
        wait_for_port(ai_url, ai_process, timeout=float(args.start_timeout))
        results = []
        for case in cases:
            print(f"\n=== Running GAIA case {case.case_id} ===\n")
            result = run_case(
                case=case,
                args=args,
                psi_root=psi_root,
                agent_root=agent_root,
                cache_root=cache_root,
                report_dir=report_dir,
                ai_url=ai_url,
                env=env,
            )
            results.append(result)
            write_report(run_id, report_dir)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status}: {case.case_id} -> {result['prediction']}")
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        state["status"] = "ok"
        state["summary"] = dict(
            Counter("passed" if item["passed"] else "failed" for item in results)
        )
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        exit_code = 1
        raise
    finally:
        stop_process(ai_process, ai_log)

    report_path = write_report(run_id, report_dir)
    print(f"\nReport: {report_path}")
    print(f"Run state: {state_path}")
    print(f"Cases: {report_dir / 'cases'}")
    return exit_code
