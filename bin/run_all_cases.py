#!/usr/bin/env python3
"""Terminal-Bench 评测主流程 — 通过 harbor run 调用 psi-agent。

harbor 管理容器生命周期、verifier 执行和 result.json 生成。
本脚本只负责：case 选择、批量调用、结果收集。

用法:
  python3 run_all_cases.py                          # 跑所有 case
  python3 run_all_cases.py --versions 2.1           # 只跑 2.1
  python3 run_all_cases.py --cases fix-git          # 指定 case
  python3 run_all_cases.py --limit 5                 # 前 5 个
  python3 run_all_cases.py --list                    # 列出，不执行
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKDIR = Path(os.environ.get(
    "TB_BENCH_WORKDIR",
    f"{os.environ.get('HOME', '/root')}/psi-agent-benchmark",
))
TASKS_DIR = WORKDIR / "tasks"
JOBS_DIR = WORKDIR / "jobs"
CONFIG_PATH = WORKDIR / "config" / "benchmark.yaml"
CASE_META = WORKDIR / "config" / "case_metadata.json"
HARBOR_BIN = os.environ.get("TB_HARBOR_BIN", "harbor")
AGENT_IMPORT = "adapters.terminal_bench.harbor_agent:PsiAgent"


def _load_env():
    """Load .env from WORKDIR so --ae flags get real values."""
    env_file = WORKDIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_env()
MODEL = os.environ.get("PSI_AI_MODEL", "glm-5.3-max")


def load_cases(versions=None, cases=None, exclude=None, difficulties=None, limit=None):
    """Load and filter cases from case_metadata.json."""
    if not CASE_META.exists():
        print(f"[ERROR] {CASE_META} not found. Run fetch_cases.py first.")
        sys.exit(1)

    with open(CASE_META, encoding="utf-8") as f:
        meta = json.load(f)

    all_cases = meta.get("cases", [])
    selected = []

    for c in all_cases:
        if not c.get("enabled", True):
            continue
        ver = c.get("version", "2.1")
        if versions and ver not in versions:
            continue
        name = c.get("name", "")
        if cases and name not in cases:
            continue
        if exclude and name in exclude:
            continue
        diff = c.get("difficulty", "")
        if difficulties and diff not in difficulties:
            continue
        selected.append(c)

    if limit:
        selected = selected[:limit]
    return selected


def run_harbor(case_name, version, run_id, jobs_dir, timeout=2400):
    """Run a single case via harbor, return the result dict."""
    dataset = f"terminal-bench@{'2.0' if version == '2.1' else '3.0'}"
    job_dir = jobs_dir / f"{run_id}" / case_name.replace("/", "_")

    cmd = [
        HARBOR_BIN, "run",
        "-d", dataset,
        "--agent-import-path", AGENT_IMPORT,
        "--model", f"openai/{MODEL}",
        "--jobs-dir", str(job_dir),
        "--n-concurrent", "1",
        "--include-task-name", case_name,
    ]

    env_keys = [
        "PSI_AI_API_KEY", "PSI_AI_BASE_URL", "PSI_AI_PROVIDER",
        "PSI_AI_MODEL", "PSI_AGENT_REPO", "PSI_AGENT_REF",
        "PSI_AGENT_WORKSPACE",
    ]
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            cmd.extend(["--ae", f"{key}={val}"])

    print(f"  [harbor] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120)

    if result.returncode != 0:
        print(f"  [harbor] FAILED (rc={result.returncode})")
        if result.stderr:
            print(f"  [harbor] stderr: {result.stderr[-500:]}")

    result_json = _find_result_json(job_dir)
    return result_json


def _find_result_json(job_dir):
    """Find the latest result.json in the job directory."""
    if not job_dir.exists():
        return None
    candidates = sorted(job_dir.rglob("result.json"), reverse=True)
    if not candidates:
        return None
    try:
        with open(candidates[0], encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Terminal-Bench via harbor")
    parser.add_argument("--versions", nargs="*", default=None,
                        help="Filter by version (2.1, 3.0)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Specific case names")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Exclude case names")
    parser.add_argument("--difficulties", nargs="*", default=None,
                        help="Filter by difficulty")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max cases to run")
    parser.add_argument("--list", action="store_true",
                        help="List cases and exit")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier for job directory")
    parser.add_argument("--timeout", type=int, default=2400,
                        help="Per-case timeout in seconds")
    args = parser.parse_args()

    cases = load_cases(
        versions=args.versions,
        cases=args.cases,
        exclude=args.exclude,
        difficulties=args.difficulties,
        limit=args.limit,
    )

    if args.list:
        print(f"{'#':>3}  {'ver':>3}  {'diff':>4}  name")
        print("-" * 50)
        for i, c in enumerate(cases, 1):
            print(f"{i:>3}  {c.get('version','?'):>3}  "
                  f"{c.get('difficulty','?'):>4}  {c['name']}")
        print(f"\nTotal: {len(cases)} cases")
        return

    if not cases:
        print("No cases matched the filters.")
        return

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    jobs_dir = JOBS_DIR
    jobs_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Terminal-Bench via harbor | run_id={run_id}")
    print(f"  Cases: {len(cases)} | Model: {MODEL}")
    print(f"  Jobs: {jobs_dir}")
    print(f"{'='*60}")

    results = []
    passed = 0
    for i, c in enumerate(cases, 1):
        name = c["name"]
        ver = c.get("version", "2.1")
        print(f"\n[{i}/{len(cases)}] {ver}/{name}")

        result = run_harbor(name, ver, run_id, jobs_dir, args.timeout)
        if result:
            reward = result.get("reward", result.get("success", 0))
            if isinstance(reward, bool):
                reward = int(reward)
            passed += 1 if reward else 0
            results.append({"case": name, "version": ver, **result})
            print(f"  reward={reward}")
        else:
            results.append({"case": name, "version": ver, "reward": None,
                            "error": "no result.json found"})
            print(f"  reward=UNKNOWN (no result.json)")

    # Save aggregated results
    summary_path = jobs_dir / f"{run_id}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "model": MODEL,
        "total": len(cases),
        "passed": passed,
        "pass_rate": f"{passed}/{len(cases)}",
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(cases)} passed ({100*passed/len(cases):.1f}%)")
    print(f"  Summary: {summary_path}")
    print(f"  Generate report: python bin/generate_report.py --run-id {run_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
