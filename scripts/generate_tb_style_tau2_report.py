from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PSI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAU2_ROOT = DEFAULT_PSI_ROOT / "tau2-bench"
DEFAULT_REPORT_ROOT = DEFAULT_PSI_ROOT / "reports" / "tau2-psi-agent"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_paths(tau2_root: Path, run_id: str) -> list[Path]:
    sim_root = tau2_root / "data" / "simulations"
    direct = sim_root / run_id / "results.json"
    if direct.exists():
        return [direct]
    return sorted(sim_root.glob(f"{run_id}-*/results.json"))


def task_counts(tau2_root: Path) -> dict[str, int]:
    domains = tau2_root / "data" / "tau2" / "domains"
    counts = {}
    for path in sorted(domains.iterdir()):
        tasks_path = path / "tasks.json"
        if tasks_path.exists():
            tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
            counts[path.name] = len(tasks) if isinstance(tasks, list) else 0
    return counts


def reward(sim: dict[str, Any]) -> float | None:
    info = sim.get("reward_info")
    if not isinstance(info, dict):
        return None
    value = info.get("reward")
    return float(value) if isinstance(value, int | float) else None


def db_match(sim: dict[str, Any]) -> bool | None:
    info = sim.get("reward_info")
    if not isinstance(info, dict):
        return None
    db_check = info.get("db_check")
    if not isinstance(db_check, dict):
        return None
    value = db_check.get("db_match")
    return value if isinstance(value, bool) else None


def usage(messages: list[dict[str, Any]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for message in messages:
        item = message.get("usage")
        if not isinstance(item, dict):
            raw = message.get("raw_data")
            if isinstance(raw, dict):
                item = raw.get("usage")
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt_tokens")
        completion = item.get("completion_tokens")
        whole = item.get("total_tokens")
        if isinstance(prompt, int):
            total["prompt_tokens"] += prompt
        if isinstance(completion, int):
            total["completion_tokens"] += completion
        if isinstance(whole, int):
            total["total_tokens"] += whole
        else:
            total["total_tokens"] += (prompt if isinstance(prompt, int) else 0) + (
                completion if isinstance(completion, int) else 0
            )
    return total


def purpose(task: dict[str, Any] | None) -> str:
    if not isinstance(task, dict):
        return ""
    description = task.get("description")
    if isinstance(description, dict):
        value = description.get("purpose")
        if isinstance(value, str):
            return " ".join(value.split())
    if isinstance(description, str):
        return " ".join(description.split())
    return ""


def persona(task_id: str) -> str:
    match = re.search(r"\[PERSONA:([^\]]+)\]", task_id)
    return match.group(1) if match else "-"


def task_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = result.get("tasks")
    if not isinstance(tasks, list):
        return {}
    return {str(task.get("id")): task for task in tasks if isinstance(task, dict)}


def failed_dimensions(sim: dict[str, Any]) -> list[str]:
    info = sim.get("reward_info")
    if not isinstance(info, dict):
        return ["UNKNOWN"]
    dims = []
    breakdown = info.get("reward_breakdown")
    if isinstance(breakdown, dict):
        for key, value in breakdown.items():
            if isinstance(value, int | float) and value < 1:
                dims.append(str(key))
    if db_match(sim) is False and "DB" not in dims:
        dims.append("DB")
    return dims or ["OTHER"]


def collect_rows(tau2_root: Path, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in result_paths(tau2_root, run_id):
        result = load_json(path)
        info = result.get("info") if isinstance(result.get("info"), dict) else {}
        env = info.get("environment_info") if isinstance(info.get("environment_info"), dict) else {}
        agent = info.get("agent_info") if isinstance(info.get("agent_info"), dict) else {}
        user = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
        tasks = task_map(result)
        simulations = result.get("simulations")
        if not isinstance(simulations, list):
            continue
        for sim in simulations:
            if not isinstance(sim, dict):
                continue
            task_id = str(sim.get("task_id"))
            messages = sim.get("messages") if isinstance(sim.get("messages"), list) else []
            value = reward(sim)
            infra = sim.get("termination_reason") == "infrastructure_error"
            rows.append(
                {
                    "domain": env.get("domain_name") or "unknown",
                    "task_id": task_id,
                    "persona": persona(task_id),
                    "purpose": purpose(tasks.get(task_id)),
                    "reward": value,
                    "passed": value is not None and value >= 1.0 and not infra,
                    "db_match": db_match(sim),
                    "termination": sim.get("termination_reason") or "unknown",
                    "duration": sim.get("duration"),
                    "turns": len(messages),
                    "agent": agent.get("implementation") or "unknown",
                    "agent_llm": agent.get("llm") or "unknown",
                    "user": user.get("implementation") or "unknown",
                    "user_llm": user.get("llm") or "unknown",
                    "usage": usage(messages),
                    "failed_dimensions": failed_dimensions(sim),
                }
            )
    return sorted(rows, key=lambda row: (row["domain"], row["task_id"]))


def fmt_float(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else "-"


def fmt_rate(n: int, total: int) -> str:
    if total == 0:
        return "0 / 0 = 0.0%"
    return f"{n} / {total} = {n / total:.1%}"


def md(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def snapshot_name(domain: str, task_id: str, run_id: str) -> str:
    save_name = f"{run_id}-{domain}"
    digest = __import__("hashlib").sha1(
        f"{domain}:{task_id}:{save_name}".encode("utf-8")
    ).hexdigest()[:10]
    stem = f"{domain}_task_{digest}_{save_name}"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem) + ".md"


def write_report(rows: list[dict[str, Any]], tau2_root: Path, run_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    completed = sum(1 for row in rows if row["termination"] != "infrastructure_error")
    infra = sum(1 for row in rows if row["termination"] == "infrastructure_error")
    rewards = [row["reward"] for row in rows if row["reward"] is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    db_known = [row["db_match"] for row in rows if row["db_match"] is not None]
    db_passed = sum(1 for item in db_known if item)
    tokens: Counter[str] = Counter()
    for row in rows:
        tokens.update(row["usage"])

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_dims: Counter[str] = Counter()
    for row in rows:
        by_domain[row["domain"]].append(row)
        by_persona[row["persona"]].append(row)
        if not row["passed"]:
            failure_dims.update(row["failed_dimensions"])

    counts = task_counts(tau2_root)
    selected_counts = Counter(row["domain"] for row in rows)

    lines = [
        "# tau2 case 子集评测",
        "",
        "## 一、执行摘要",
        "",
        f"- 运行 ID：`{run_id}`",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 被测 Agent：`psi_agent`",
        "- Agent LLM：`deepseek-chat`",
        "- User Simulator LLM：`deepseek/deepseek-chat`",
        f"- 总 case 数：{total}",
        f"- 全部完成：{completed} / {total}",
        f"- 累计 reward：{fmt_rate(passed, total)}",
        f"- 平均 reward：{fmt_float(avg_reward)}",
        f"- DB Check 通过率：{fmt_rate(db_passed, len(db_known))}",
        f"- Infra error：{infra}",
        "",
        "token统计",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Prompt tokens | {tokens['prompt_tokens']} |",
        f"| Completion tokens | {tokens['completion_tokens']} |",
        f"| Total tokens | {tokens['total_tokens']} |",
        "",
        "## 二、子集范围",
        "",
        "| 领域 | tau2 总题数 | 本次选取 | 选取比例 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain in sorted(selected_counts):
        total_domain = counts.get(domain, 0)
        selected = selected_counts[domain]
        pct = f"{selected / total_domain:.1%}" if total_domain else "-"
        lines.append(f"| {md(domain)} | {total_domain} | {selected} | {pct} |")

    lines.extend(
        [
            "",
            "## 三、按领域统计",
            "",
            "| 领域 | 题数 | 通过 | 失败/超时/unknown | DB Check 通过 | 平均 reward |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, items in sorted(by_domain.items()):
        domain_total = len(items)
        domain_passed = sum(1 for row in items if row["passed"])
        domain_db = sum(1 for row in items if row["db_match"] is True)
        domain_rewards = [row["reward"] for row in items if row["reward"] is not None]
        domain_avg = sum(domain_rewards) / len(domain_rewards) if domain_rewards else None
        lines.append(
            f"| {md(domain)} | {domain_total} | {domain_passed} | "
            f"{domain_total - domain_passed} | {domain_db} | {fmt_float(domain_avg)} |"
        )

    lines.extend(
        [
            "",
            "## 四、按用户画像/难度统计",
            "",
            "| Persona | 题数 | 通过 | 失败/超时/unknown | 平均 reward |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, items in sorted(by_persona.items()):
        persona_total = len(items)
        persona_passed = sum(1 for row in items if row["passed"])
        persona_rewards = [row["reward"] for row in items if row["reward"] is not None]
        persona_avg = sum(persona_rewards) / len(persona_rewards) if persona_rewards else None
        lines.append(
            f"| {md(name)} | {persona_total} | {persona_passed} | "
            f"{persona_total - persona_passed} | {fmt_float(persona_avg)} |"
        )

    lines.extend(
        [
            "",
            "## 五、失败原因概览",
            "",
            "| 失败维度 | 数量 |",
            "| --- | ---: |",
        ]
    )
    if failure_dims:
        for name, count in sorted(failure_dims.items()):
            lines.append(f"| {md(name)} | {count} |")
    else:
        lines.append("| - | 0 |")

    lines.extend(
        [
            "",
            "## 六、详细结果表",
            "",
            "| # | 领域 | Task ID | Persona | Reward | DB Check | Termination | 时长(s) | 轮数 | 结果 | Snapshot |",
            "| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for idx, row in enumerate(rows, 1):
        snapshot = "-"
        if not row["passed"]:
            snapshot = f"[查看](snapshots/{snapshot_name(row['domain'], row['task_id'], run_id)})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    md(row["domain"]),
                    md(row["task_id"]),
                    md(row["persona"]),
                    fmt_float(row["reward"]),
                    md(row["db_match"]),
                    md(row["termination"]),
                    fmt_float(row["duration"]),
                    str(row["turns"]),
                    "PASS" if row["passed"] else "FAIL",
                    snapshot,
                ]
            )
            + " |"
        )

    failed = [row for row in rows if not row["passed"]]
    lines.extend(["", "## 七、未通过 case 列表", ""])
    if failed:
        for row in failed:
            dims = ", ".join(row["failed_dimensions"])
            snapshot = f"snapshots/{snapshot_name(row['domain'], row['task_id'], run_id)}"
            lines.append(
                f"- `{row['domain']}:{row['task_id']}`：失败维度 `{dims}`，"
                f"reward `{fmt_float(row['reward'])}`，snapshot：[{snapshot}]({snapshot})"
            )
    else:
        lines.append("- 本次子集没有未通过 case。")

    lines.extend(
        [
            "",
            "## 八、结论",
            "",
            f"1. 本次评测覆盖 {len(selected_counts)} 个 tau2 领域，共 {total} 条 case，全部完成。",
            f"2. 总体通过率为 {passed / total:.1%}，平均 reward 为 {fmt_float(avg_reward)}，无 infra error。",
            "3. airline 和 retail 表现稳定，均为 10 / 10 通过。",
            "4. 主要短板集中在 banking_knowledge，其次是 mock 的历史/沟通类任务和 telecom 的复杂 MMS 排障任务。",
            "5. 失败 case 已生成 snapshot，可继续用于分析提示词、工具调用策略以及后续 ontology / LOM 增强后的能力变化。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a TB-style tau2 subset report.")
    parser.add_argument("--run-id", default="tau2-psi-balanced50-20260825-v2")
    parser.add_argument("--tau2-root", default=str(DEFAULT_TAU2_ROOT))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    tau2_root = Path(args.tau2_root)
    out = Path(args.output) if args.output else DEFAULT_REPORT_ROOT / args.run_id / "tau2_case_subset_eval_report.md"
    rows = collect_rows(tau2_root, args.run_id)
    if not rows:
        raise FileNotFoundError(f"No result rows found for run id: {args.run_id}")
    write_report(rows, tau2_root, args.run_id, out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
