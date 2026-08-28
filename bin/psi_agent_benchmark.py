from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import gaia_benchmark


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSI_ROOT = Path(os.environ.get("PSI_AGENT_REPO_ROOT", BENCHMARK_ROOT.parent))
DEFAULT_TAU2_ROOT = Path(os.environ.get("TAU2_ROOT", DEFAULT_PSI_ROOT / "tau2-bench"))
DEFAULT_CONFIG = BENCHMARK_ROOT / "config" / "tau2_subsets.json"
DEFAULT_REPORT_ROOT = BENCHMARK_ROOT / "reports" / "tau2-psi-agent"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_api_key(path: Path) -> str:
    if os.environ.get("PSI_AI_API_KEY"):
        return os.environ["PSI_AI_API_KEY"]
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    text = path.read_text(encoding="utf-8").strip()
    if "=" in text:
        return text.split("=", 1)[1].strip().strip('"').strip("'")
    return text


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def cache_env(cache_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PSI_TAU2_CACHE_ROOT"] = str(cache_root)
    env["PSI_APPDATA"] = str(cache_root / "psi-appdata")
    env["UV_CACHE_DIR"] = str(cache_root / "uv")
    env["PIP_CACHE_DIR"] = str(cache_root / "pip")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env["PYTHONPYCACHEPREFIX"] = str(cache_root / "pycache")
    env["HF_HOME"] = str(cache_root / "huggingface")
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.1.0"
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PSI_AGENT"] = "0.1.0"
    return env


def wait_for_port(url: str, process: subprocess.Popen, timeout: float) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        raise ValueError(f"URL has no port: {url}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("psi-agent AI process exited before it was ready")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"psi-agent AI did not listen within {timeout}s")


def check_ai_upstream(url: str, timeout: float) -> None:
    payload = json.dumps(
        {"messages": [{"role": "user", "content": "Reply with OK only."}], "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"AI upstream health check failed: HTTP {response.status}")


def install_adapter(psi_root: Path, tau2_root: Path) -> None:
    src = BENCHMARK_ROOT / "adapters" / "tau2" / "psi_agent_adapter.py"
    dst = tau2_root / "src" / "tau2" / "agent" / "psi_agent_adapter.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    registry = tau2_root / "src" / "tau2" / "registry.py"
    text = registry.read_text(encoding="utf-8")
    if "from tau2.agent.psi_agent_adapter import create_psi_agent" not in text:
        marker = "from tau2.agent.llm_agent import ("
        idx = text.find(marker)
        if idx == -1:
            raise RuntimeError("Could not find tau2 agent import block in registry.py")
        text = (
            text[:idx]
            + "from tau2.agent.psi_agent_adapter import create_psi_agent\n"
            + text[idx:]
        )
    register_line = '    registry.register_agent_factory(create_psi_agent, "psi_agent")\n'
    if register_line not in text:
        marker = "    registry.register_agent_factory(\n        create_discrete_time_audio_native_agent,"
        idx = text.find(marker)
        if idx == -1:
            raise RuntimeError("Could not find tau2 agent registration block in registry.py")
        end = text.find("    )\n", idx)
        if end == -1:
            raise RuntimeError("Could not find end of tau2 registration block")
        end += len("    )\n")
        text = text[:end] + register_line + text[end:]
    registry.write_text(text, encoding="utf-8")

    workspace = psi_root / "examples" / "tau2-workspace" / "systems"
    workspace.mkdir(parents=True, exist_ok=True)
    system_py = workspace / "system.py"
    if not system_py.exists():
        system_py.write_text(
            (
                "async def system_prompt_builder(context):\n"
                "    return '''You are connected to tau2-bench through an adapter.\n"
                "Return strict JSON only.\n"
                "Use {\"message\": \"...\"} for user-facing replies.\n"
                "Use {\"tool_calls\": [{\"name\": \"tool_name\", \"arguments\": {}}]} for tools.\n"
                "Do not wrap JSON in markdown.'''\n"
            ),
            encoding="utf-8",
        )


def load_subset(config_path: Path, subset_name: str) -> tuple[str, dict[str, list[str]], str]:
    config = load_json(config_path)
    if not subset_name:
        subset_name = str(config.get("default_subset") or "")
    subsets = config.get("subsets")
    if not isinstance(subsets, dict) or subset_name not in subsets:
        raise KeyError(f"Subset {subset_name!r} not found in {config_path}")
    subset = subsets[subset_name]
    if not isinstance(subset, dict):
        raise ValueError(f"Subset {subset_name!r} is not an object")
    cases = subset.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"Subset {subset_name!r} has no cases object")
    normalized = {
        str(domain): [str(task_id) for task_id in task_ids]
        for domain, task_ids in cases.items()
        if isinstance(task_ids, list) and task_ids
    }
    return subset_name, normalized, str(subset.get("description") or "")


def valid_task_ids(tau2_root: Path, domain: str) -> set[str]:
    tasks_path = tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        return set()
    return {str(task.get("id")) for task in tasks if isinstance(task, dict)}


def available_domains(tau2_root: Path) -> list[str]:
    domains_root = tau2_root / "data" / "tau2" / "domains"
    return sorted(path.name for path in domains_root.iterdir() if (path / "tasks.json").exists())


def first_task_ids(tau2_root: Path, domain: str, limit: int) -> list[str]:
    tasks_path = tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        return []
    ids = [str(task.get("id")) for task in tasks if isinstance(task, dict) and task.get("id")]
    return ids[:limit]


def select_cases(args: argparse.Namespace, tau2_root: Path) -> tuple[str, dict[str, list[str]], str]:
    if args.domains or args.limit_per_domain:
        domains = args.domains or available_domains(tau2_root)
        limit = int(args.limit_per_domain or 10)
        if limit < 1:
            raise ValueError("--limit-per-domain must be greater than 0")
        cases = {domain: first_task_ids(tau2_root, domain, limit) for domain in domains}
        empty = [domain for domain, task_ids in cases.items() if not task_ids]
        if empty:
            raise ValueError("No task ids found for domain(s): " + ", ".join(empty))
        subset_name = "custom_" + "_".join(domains) + f"_{limit}"
        description = f"Dynamic subset: first {limit} case(s) from {', '.join(domains)}."
        return subset_name, cases, description
    return load_subset(Path(args.subset_file), args.subset)


def validate_subset(tau2_root: Path, cases: dict[str, list[str]]) -> None:
    errors = []
    for domain, task_ids in cases.items():
        ids = valid_task_ids(tau2_root, domain)
        missing = [task_id for task_id in task_ids if task_id not in ids]
        if missing:
            errors.append(f"{domain}: {', '.join(missing)}")
    if errors:
        raise ValueError("Invalid tau2 task ids in subset: " + "; ".join(errors))


def result_paths_for_run(tau2_root: Path, run_id: str) -> list[Path]:
    sim_root = tau2_root / "data" / "simulations"
    direct = sim_root / run_id / "results.json"
    if direct.exists():
        return [direct]
    return sorted(sim_root.glob(f"{run_id}-*/results.json"))


def reward_value(sim: dict[str, Any]) -> float | None:
    reward_info = sim.get("reward_info")
    if not isinstance(reward_info, dict):
        return None
    reward = reward_info.get("reward")
    return float(reward) if isinstance(reward, int | float) else None


def db_match_value(sim: dict[str, Any]) -> bool | None:
    reward_info = sim.get("reward_info")
    if not isinstance(reward_info, dict):
        return None
    db_check = reward_info.get("db_check")
    if not isinstance(db_check, dict):
        return None
    db_match = db_check.get("db_match")
    return db_match if isinstance(db_match, bool) else None


def usage_totals(messages: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for message in messages:
        usage = message.get("usage")
        if not isinstance(usage, dict):
            raw_data = message.get("raw_data")
            if isinstance(raw_data, dict):
                usage = raw_data.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if isinstance(prompt, int):
            totals["prompt_tokens"] += prompt
        if isinstance(completion, int):
            totals["completion_tokens"] += completion
        if isinstance(total, int):
            totals["total_tokens"] += total
        else:
            totals["total_tokens"] += (prompt if isinstance(prompt, int) else 0) + (
                completion if isinstance(completion, int) else 0
            )
    return totals


def task_map(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(task.get("id")): task for task in tasks if isinstance(task, dict)}


def task_purpose(task: dict[str, Any] | None) -> str:
    if not task:
        return ""
    description = task.get("description")
    if isinstance(description, dict):
        purpose = description.get("purpose")
        if isinstance(purpose, str):
            return " ".join(purpose.split())
    if isinstance(description, str):
        return " ".join(description.split())
    return ""


def result_row(path: Path, result: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    info = result.get("info") if isinstance(result.get("info"), dict) else {}
    env_info = info.get("environment_info") if isinstance(info.get("environment_info"), dict) else {}
    agent_info = info.get("agent_info") if isinstance(info.get("agent_info"), dict) else {}
    user_info = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
    tasks = task_map(result.get("tasks") if isinstance(result.get("tasks"), list) else [])
    task_id = str(sim.get("task_id"))
    messages = sim.get("messages") if isinstance(sim.get("messages"), list) else []
    reward = reward_value(sim)
    infra_error = sim.get("termination_reason") == "infrastructure_error"
    return {
        "source": str(path),
        "save_name": path.parent.name,
        "domain": env_info.get("domain_name") or info.get("domain") or "unknown",
        "agent": agent_info.get("implementation") or "unknown",
        "agent_llm": agent_info.get("llm") or "unknown",
        "user": user_info.get("implementation") or "unknown",
        "user_llm": user_info.get("llm") or "unknown",
        "task_id": task_id,
        "purpose": task_purpose(tasks.get(task_id)),
        "reward": reward,
        "passed": reward is not None and reward >= 1.0 and not infra_error,
        "db_match": db_match_value(sim),
        "termination": sim.get("termination_reason") or "unknown",
        "duration": sim.get("duration"),
        "turns": len(messages),
        "agent_cost": sim.get("agent_cost"),
        "user_cost": sim.get("user_cost"),
        "usage": usage_totals(messages),
        "simulation": sim,
        "task": tasks.get(task_id),
    }


def collect_rows(tau2_root: Path, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in result_paths_for_run(tau2_root, run_id):
        result = load_json(path)
        simulations = result.get("simulations")
        if not isinstance(simulations, list):
            continue
        for sim in simulations:
            if isinstance(sim, dict):
                rows.append(result_row(path, result, sim))
    rows.sort(key=lambda r: (str(r["domain"]), str(r["task_id"]), str(r["save_name"])))
    return rows


def fmt_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0 / 0 = 0.0%"
    return f"{numerator} / {denominator} = {numerator / denominator:.1%}"


def fmt_float(value: object, default: str = "-") -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else default


def md_escape(value: object) -> str:
    return ("" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")


def conversation_markdown(messages: list[dict[str, Any]]) -> str:
    lines = []
    for idx, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        lines.append(f"### Turn {idx}: {role}")
        if content:
            lines.append(str(content).strip())
        if tool_calls:
            lines.append("```json")
            lines.append(json.dumps(tool_calls, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_snapshots(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for row in rows:
        if row["passed"]:
            continue
        digest = hashlib.sha1(
            f"{row['domain']}:{row['task_id']}:{row['save_name']}".encode("utf-8")
        ).hexdigest()[:10]
        stem = f"{row['domain']}_task_{digest}_{row['save_name']}"
        stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
        json_path = snapshot_dir / f"{stem}.json"
        md_path = snapshot_dir / f"{stem}.md"
        payload = {
            "domain": row["domain"],
            "task_id": row["task_id"],
            "purpose": row["purpose"],
            "passed": row["passed"],
            "reward": row["reward"],
            "db_match": row["db_match"],
            "termination": row["termination"],
            "duration": row["duration"],
            "turns": row["turns"],
            "source": row["source"],
            "task": row["task"],
            "simulation": row["simulation"],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        messages = row["simulation"].get("messages")
        if not isinstance(messages, list):
            messages = []
        md_path.write_text(
            "\n".join(
                [
                    f"# Failed Case Snapshot: {row['domain']} / task {row['task_id']}",
                    "",
                    f"- Reward: {fmt_float(row['reward'])}",
                    f"- DB Match: {row['db_match']}",
                    f"- Termination: {row['termination']}",
                    f"- Duration: {fmt_float(row['duration'])} s",
                    f"- Turns: {row['turns']}",
                    f"- Source: `{row['source']}`",
                    "",
                    "## Task Purpose",
                    row["purpose"] or "-",
                    "",
                    "## Conversation",
                    conversation_markdown(messages),
                ]
            ),
            encoding="utf-8",
        )
        index[f"{row['domain']}:{row['task_id']}"] = str(md_path)
    return index


def group_by_domain(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    grouped = {}
    for domain, domain_rows in by_domain.items():
        total = len(domain_rows)
        passed = sum(1 for row in domain_rows if row["passed"])
        infra = sum(1 for row in domain_rows if row["termination"] == "infrastructure_error")
        rewards = [row["reward"] for row in domain_rows if row["reward"] is not None]
        grouped[domain] = {
            "total": total,
            "passed": passed,
            "failed_or_unknown": total - passed,
            "infra_errors": infra,
            "avg_reward": sum(rewards) / len(rewards) if rewards else None,
        }
    return grouped


def write_report(tau2_root: Path, run_id: str, out_dir: Path) -> Path:
    rows = collect_rows(tau2_root, run_id)
    if not rows:
        raise FileNotFoundError(f"No tau2 result files found for run id: {run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = result_paths_for_run(tau2_root, run_id)
    (out_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "result_files": [str(path) for path in paths]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    completed = sum(1 for row in rows if row["termination"] != "infrastructure_error")
    infra = sum(1 for row in rows if row["termination"] == "infrastructure_error")
    rewards = [row["reward"] for row in rows if row["reward"] is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    db_known = [row["db_match"] for row in rows if row["db_match"] is not None]
    db_passed = sum(1 for value in db_known if value)
    usage = Counter()
    for row in rows:
        usage.update(row["usage"])
    snapshot_index = write_snapshots(rows, out_dir)
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": total,
        "completed_cases": completed,
        "passed_cases": passed,
        "pass_rate": passed / total if total else 0,
        "average_reward": avg_reward,
        "infra_errors": infra,
        "db_passed": db_passed,
        "db_known": len(db_known),
        "usage": dict(usage),
        "by_domain": group_by_domain(rows),
        "snapshots": snapshot_index,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# tau2 case 子集评测报告",
        "",
        f"- 运行 ID: `{run_id}`",
        f"- 生成时间: {summary['generated_at']}",
        "- 被测 Agent: `psi_agent`",
        "",
        "## 一、执行摘要",
        "",
        f"- 总 case 数: {total}",
        f"- 完成 case 数: {completed} / {total}",
        f"- 通过率: {fmt_rate(passed, total)}",
        f"- 平均 reward: {fmt_float(avg_reward)}",
        f"- DB Check 通过率: {fmt_rate(db_passed, len(db_known))}",
        f"- Infra error: {infra}",
        "",
        "## 二、Token 统计",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Prompt tokens | {usage['prompt_tokens']} |",
        f"| Completion tokens | {usage['completion_tokens']} |",
        f"| Total tokens | {usage['total_tokens']} |",
        "",
        "## 三、按领域统计",
        "",
        "| 领域 | 题数 | 通过 | 失败/unknown | Infra error | 平均 reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, item in sorted(summary["by_domain"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(domain),
                    str(item["total"]),
                    str(item["passed"]),
                    str(item["failed_or_unknown"]),
                    str(item["infra_errors"]),
                    fmt_float(item["avg_reward"]),
                ]
            )
            + " |"
        )

    termination_counts = Counter(str(row["termination"]) for row in rows)
    lines.extend(["", "## 四、按终止原因统计", "", "| 终止原因 | 数量 |", "| --- | ---: |"])
    for reason, count in sorted(termination_counts.items()):
        lines.append(f"| {md_escape(reason)} | {count} |")

    lines.extend(
        [
            "",
            "## 五、详细结果表",
            "",
            "| 领域 | Task ID | Reward | DB Check | Termination | 时长(s) | 轮数 | 结果 | Snapshot |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        key = f"{row['domain']}:{row['task_id']}"
        snapshot = snapshot_index.get(key)
        snapshot_link = "-"
        if snapshot:
            rel = Path(snapshot).relative_to(out_dir).as_posix()
            snapshot_link = f"[查看]({rel})"
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row["domain"]),
                    md_escape(row["task_id"]),
                    fmt_float(row["reward"]),
                    md_escape(row["db_match"]),
                    md_escape(row["termination"]),
                    fmt_float(row["duration"]),
                    str(row["turns"]),
                    "PASS" if row["passed"] else "FAIL",
                    snapshot_link,
                ]
            )
            + " |"
        )

    lines.extend(["", "## 六、失败 Case 快照", ""])
    if snapshot_index:
        for key, snapshot in sorted(snapshot_index.items()):
            rel = Path(snapshot).relative_to(out_dir).as_posix()
            lines.append(f"- `{key}`: [{rel}]({rel})")
    else:
        lines.append("- 本次子集没有失败 case。")

    conclusion = "本次子集全部通过，可以继续扩大样本或加入 LOM/ontology 版本做对照。"
    if infra:
        conclusion = "存在 infra error，优先查看 snapshots 和运行日志，排除环境/API/进程问题。"
    elif passed < total:
        conclusion = "存在未通过 case，建议优先分析失败快照中的对话轨迹和工具调用。"
    lines.extend(
        [
            "",
            "## 七、关键观察与结论",
            "",
            f"1. 本次评测覆盖 {total} 个 tau2 case，通过 {passed} 个。",
            f"2. 平均 reward 为 {fmt_float(avg_reward)}，DB Check 通过 {db_passed} / {len(db_known)}。",
            f"3. {conclusion}",
            "",
        ]
    )
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_tau2_domain(
    *,
    uv: str,
    tau2_root: Path,
    domain: str,
    task_ids: list[str],
    save_to: str,
    model: str,
    user_llm: str,
    max_concurrency: str,
    max_retries: str,
    timeout: str,
    retrieval_config: str,
    env: dict[str, str],
    log_path: Path,
) -> int:
    command = [
        uv,
        "run",
        "--no-sync",
        "python",
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        domain,
        "--agent",
        "psi_agent",
        "--agent-llm",
        model,
        "--user-llm",
        user_llm,
        "--task-ids",
        *task_ids,
        "--save-to",
        save_to,
        "--max-concurrency",
        max_concurrency,
        "--max-retries",
        max_retries,
        "--auto-resume",
    ]
    if timeout:
        command.extend(["--timeout", timeout])
    if retrieval_config:
        command.extend(["--retrieval-config", retrieval_config])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(tau2_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def command_run(args: argparse.Namespace) -> int:
    if args.benchmark == "gaia":
        return gaia_benchmark.command_run(args)

    psi_root = Path(args.psi_root).resolve()
    tau2_root = Path(args.tau2_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    report_root = Path(args.report_root).resolve()

    if args.install_adapter:
        install_adapter(psi_root, tau2_root)
    subset_name, cases, description = select_cases(args, tau2_root)
    validate_subset(tau2_root, cases)
    run_id = args.run_id or f"tau2-psi-{subset_name}-{datetime.now():%Y%m%d-%H%M%S}"
    report_dir = report_root / run_id
    log_dir = report_dir / "logs"
    total_cases = sum(len(task_ids) for task_ids in cases.values())

    print(f"Subset: {subset_name} ({total_cases} cases)")
    for domain, task_ids in cases.items():
        print(f"- {domain}: {len(task_ids)} cases")
        if args.dry_run:
            for task_id in task_ids:
                print(f"  - {task_id}")
    if args.dry_run:
        print("\nDry run only; no API calls were made.")
        return 0

    cache_root.mkdir(parents=True, exist_ok=True)
    env = cache_env(cache_root)
    tau2_src = str(tau2_root / "src")
    env["PYTHONPATH"] = tau2_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PSI_AGENT_REPO_ROOT"] = str(psi_root)
    env["TAU2_ROOT"] = str(tau2_root)
    env["PSI_TAU2_WORKSPACE"] = str(psi_root / "examples" / "tau2-workspace")
    env["PSI_TAU2_UV"] = args.uv
    env["PSI_AI_PROVIDER"] = args.provider
    env["PSI_AI_MODEL"] = args.model
    env["PSI_AI_BASE_URL"] = args.base_url
    env["PSI_AI_API_KEY"] = env.get("PSI_AI_API_KEY") or read_api_key(Path(args.api_key_file))
    env.setdefault("DEEPSEEK_API_KEY", env["PSI_AI_API_KEY"])
    env.setdefault("TAU2_LLM_NL_ASSERTIONS", args.user_llm)
    env.setdefault("TAU2_LLM_ENV_INTERFACE", args.user_llm)

    ai_url = f"http://127.0.0.1:{free_port()}"
    env["PSI_TAU2_AI_SOCKET"] = ai_url
    state = {
        "run_id": run_id,
        "subset": subset_name,
        "description": description,
        "psi_root": str(psi_root),
        "tau2_root": str(tau2_root),
        "cache_root": str(cache_root),
        "ai_url": ai_url,
        "model": args.model,
        "user_llm": args.user_llm,
        "cases": cases,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "domains": {},
    }
    state_path = report_dir / "run_state.json"
    write_state(state_path, state)

    log_dir.mkdir(parents=True, exist_ok=True)
    ai_log_file = (log_dir / "psi-ai.log").open("w", encoding="utf-8")
    ai_process = subprocess.Popen(
        [
            args.uv,
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
        stdout=ai_log_file,
        stderr=subprocess.STDOUT,
    )
    exit_code = 0
    try:
        wait_for_port(ai_url, ai_process, timeout=45)
        if not args.skip_ai_health_check:
            check_ai_upstream(ai_url, timeout=60)
        for domain, task_ids in cases.items():
            save_to = f"{run_id}-{domain}"
            state["domains"][domain] = {
                "task_ids": task_ids,
                "save_to": save_to,
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_state(state_path, state)
            print(f"\n=== Running {domain}: {len(task_ids)} cases ===\n")
            code = run_tau2_domain(
                uv=args.uv,
                tau2_root=tau2_root,
                domain=domain,
                task_ids=task_ids,
                save_to=save_to,
                model=args.model,
                user_llm=args.user_llm,
                max_concurrency=args.max_concurrency,
                max_retries=args.max_retries,
                timeout=args.timeout,
                retrieval_config=args.banking_retrieval_config
                if domain == "banking_knowledge"
                else "",
                env=env,
                log_path=log_dir / f"{domain}.log",
            )
            state["domains"][domain]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            state["domains"][domain]["status"] = "ok" if code == 0 else "failed"
            state["domains"][domain]["exit_code"] = code
            write_state(state_path, state)
            if result_paths_for_run(tau2_root, run_id):
                write_report(tau2_root, run_id, report_dir)
            if code != 0:
                exit_code = code
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        state["status"] = "ok" if exit_code == 0 else "failed"
        write_state(state_path, state)
        report_path = write_report(tau2_root, run_id, report_dir)
    finally:
        ai_process.terminate()
        try:
            ai_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ai_process.kill()
            ai_process.wait(timeout=10)
        ai_log_file.close()
    print(f"\nReport: {report_path}")
    print(f"Run state: {state_path}")
    print(f"Snapshots: {report_dir / 'snapshots'}")
    return exit_code


def command_report(args: argparse.Namespace) -> int:
    if args.benchmark == "gaia":
        return gaia_benchmark.command_report(args)

    tau2_root = Path(args.tau2_root).resolve()
    report_root = Path(args.report_root).resolve()
    report_path = write_report(tau2_root, args.run_id, report_root / args.run_id)
    print(report_path)
    return 0


def command_list_subsets(args: argparse.Namespace) -> int:
    if args.benchmark == "gaia":
        return gaia_benchmark.command_list_subsets(args)

    config = load_json(Path(args.subset_file))
    subsets = config.get("subsets")
    if not isinstance(subsets, dict):
        raise ValueError("No subsets object found")
    for name, subset in subsets.items():
        cases = subset.get("cases") if isinstance(subset, dict) else {}
        total = sum(len(v) for v in cases.values()) if isinstance(cases, dict) else 0
        description = subset.get("description") if isinstance(subset, dict) else ""
        print(f"{name}: {total} cases")
        if description:
            print(f"  {description}")
        if isinstance(cases, dict):
            for domain, task_ids in cases.items():
                print(f"  - {domain}: {len(task_ids)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="psi-agent tau2 benchmark automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--benchmark", choices=["tau2", "gaia"], default="tau2")
    common.add_argument("--psi-root", default=str(DEFAULT_PSI_ROOT))
    common.add_argument("--tau2-root", default=str(DEFAULT_TAU2_ROOT))
    common.add_argument("--subset-file", default=str(DEFAULT_CONFIG))
    common.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--subset", default="")
    run_parser.add_argument(
        "--domains",
        nargs="*",
        default=[],
        help="Build a temporary subset from these tau2 domains instead of a named subset.",
    )
    run_parser.add_argument(
        "--limit-per-domain",
        type=int,
        default=0,
        help="Build a temporary subset with the first N task ids from each selected domain.",
    )
    run_parser.add_argument("--run-id", default="")
    run_parser.add_argument("--uv", default=os.environ.get("PSI_TAU2_UV", "uv"))
    run_parser.add_argument("--provider", default=os.environ.get("PSI_AI_PROVIDER", "openai"))
    run_parser.add_argument("--model", default=os.environ.get("PSI_AI_MODEL", "glm-5.3-max"))
    run_parser.add_argument(
        "--base-url",
        default=os.environ.get("PSI_AI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    )
    run_parser.add_argument("--api-key-file", default=str(DEFAULT_TAU2_ROOT / "api"))
    run_parser.add_argument("--user-llm", default=os.environ.get("TAU2_USER_LLM", "glm-5.3-max"))
    run_parser.add_argument("--max-concurrency", default="1")
    run_parser.add_argument("--max-retries", default="0")
    run_parser.add_argument("--timeout", default="600")
    run_parser.add_argument("--cache-root", default=str(DEFAULT_PSI_ROOT / ".cache" / "tau2-benchmark"))
    run_parser.add_argument("--banking-retrieval-config", default="full_kb")
    run_parser.add_argument("--skip-ai-health-check", action="store_true")
    run_parser.add_argument("--install-adapter", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--gaia-subset-file", default=str(gaia_benchmark.DEFAULT_GAIA_CONFIG))
    run_parser.add_argument("--gaia-data-root", default=str(gaia_benchmark.DEFAULT_GAIA_DATA_ROOT))
    run_parser.add_argument("--gaia-hf-subset", default="")
    run_parser.add_argument("--gaia-split", default="")
    run_parser.add_argument("--gaia-agent-root", default="")
    run_parser.add_argument("--instance-ids", nargs="*", default=[])
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--start-timeout", default="45")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected subset and exit before starting psi-agent or tau2.",
    )
    run_parser.set_defaults(func=command_run)

    report_parser = subparsers.add_parser("report", parents=[common])
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--gaia-subset-file", default=str(gaia_benchmark.DEFAULT_GAIA_CONFIG))
    report_parser.set_defaults(func=command_report)

    install_parser = subparsers.add_parser("install-adapter", parents=[common])
    install_parser.set_defaults(
        func=lambda args: (install_adapter(Path(args.psi_root), Path(args.tau2_root)) or 0)
    )

    list_parser = subparsers.add_parser("list-subsets", parents=[common])
    list_parser.add_argument("--gaia-subset-file", default=str(gaia_benchmark.DEFAULT_GAIA_CONFIG))
    list_parser.set_defaults(func=command_list_subsets)

    download_gaia_parser = subparsers.add_parser("download-gaia")
    download_gaia_parser.add_argument(
        "--gaia-data-root", default=str(gaia_benchmark.DEFAULT_GAIA_DATA_ROOT)
    )
    download_gaia_parser.add_argument("--gaia-repo-id", default=gaia_benchmark.DEFAULT_GAIA_REPO_ID)
    download_gaia_parser.add_argument("--gaia-revision", default="main")
    download_gaia_parser.set_defaults(func=gaia_benchmark.command_download)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, KeyError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
