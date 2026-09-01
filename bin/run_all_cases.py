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
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HARBOR_BIN = os.environ.get("TB_HARBOR_BIN", "harbor")
AGENT_IMPORT = "adapters.terminal_bench.harbor_agent:PsiAgent"


def _load_env(path):
    """Load .env from given path, setting env vars with setdefault."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# Bootstrap: load .env from cwd or known locations before computing WORKDIR
for _candidate in [Path.cwd() / ".env",
                   Path.home() / "psi-agent-bench-v2" / ".env",
                   Path.home() / "psi-agent-benchmark" / ".env"]:
    if _candidate.exists():
        _load_env(_candidate)
        break

WORKDIR = Path(os.environ.get(
    "TB_BENCH_WORKDIR",
    f"{os.environ.get('HOME', '/root')}/psi-agent-benchmark",
))
TASKS_DIR = WORKDIR / "tasks"
JOBS_DIR = WORKDIR / "jobs"
CONFIG_PATH = WORKDIR / "config" / "benchmark.yaml"
CASE_META = WORKDIR / "config" / "case_metadata.json"
MODEL = os.environ.get("PSI_AI_MODEL", "glm-5.3-max")


def load_cases(versions=None, cases=None, exclude=None, difficulties=None, limit=None):
    """Load and filter cases from case_metadata.json."""
    if not CASE_META.exists():
        print(f"[ERROR] {CASE_META} not found. Run bin/fetch_cases.py first.")
        sys.exit(1)

    with open(CASE_META, encoding="utf-8") as f:
        meta = json.load(f)

    # case_metadata.json is { "version/name": {name, version, ...}, ... }
    if isinstance(meta, dict):
        all_cases = list(meta.values())
    else:
        all_cases = meta.get("cases", [])

    selected = []
    for c in all_cases:
        # Skip disabled cases only when not targeting specific cases
        if not cases and not c.get("enabled", True):
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


def run_harbor(case_name, version, run_id, jobs_dir, timeout=18000):
    """Run a single case via harbor, return the result dict."""
    if version == "2.1":
        dataset = "terminal-bench@2.0"
    else:
        # TB3 的正确注册标识（legacy terminal-bench@3.0 不存在于注册表）
        dataset = "terminal-bench/terminal-bench@3.0.0"
    job_dir = jobs_dir / f"{run_id}" / case_name.replace("/", "_")

    cmd = [
        HARBOR_BIN, "run",
        "-d", dataset,
        "--agent-import-path", AGENT_IMPORT,
        "--model", f"openai/{MODEL}",
        "--jobs-dir", str(job_dir),
        "--n-concurrent", "1",
        "--include-task-name", (f"terminal-bench/{case_name}" if version == "3.0" else case_name),
    ]

    env_keys = [
        "PSI_AI_API_KEY", "PSI_AI_BASE_URL", "PSI_AI_PROVIDER",
        "PSI_AI_MODEL", "PSI_AGENT_REPO", "PSI_AGENT_REF",
        "PSI_AGENT_WORKSPACE",
        "PSI_AI_REASONING_EFFORT",
    ]
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            cmd.extend(["--ae", f"{key}={val}"])
    # 让 adapter 知道当前 case 名（用于保存 ai.log 到 pilot_results）
    cmd.extend(["--ae", f"PSI_AGENT_CASE={case_name}"])

    print(f"  [harbor] {' '.join(cmd)}")
    # 让 harbor 子进程（含 adapter）知道当前 case 名，用于保存 ai.log 到 pilot_results
    os.environ["PSI_AGENT_CASE"] = case_name
    # Harbor subprocess needs repo root in PYTHONPATH for adapter imports
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    # Popen + start_new_session: 让 harbor 成为进程组领导，超时可用 killpg 杀整棵进程树，
    # 避免 subprocess.run 只杀直接子进程而留下孤儿 docker 容器继续烧额度。
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout + 120)
    except subprocess.TimeoutExpired:
        # 安全网触发：杀整个进程组 + 清理该 case 的容器（best-effort），不崩溃、继续后续 case
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
        _stop_case_containers(case_name)
        print(f"  [harbor] TIMEOUT after {timeout}s (safety net) — killed process group and containers")
        return None

    if proc.returncode != 0:
        print(f"  [harbor] FAILED (rc={proc.returncode})")
        if stderr:
            print(f"  [harbor] stderr: {stderr[-500:]}")

    result_json = _find_result_json(job_dir)
    return result_json


def _stop_case_containers(case_name):
    """Best-effort kill any running containers whose name matches the case."""
    try:
        ps = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        )
        for name in ps.stdout.splitlines():
            if case_name in name:
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
                print(f"  [harbor] killed orphan container: {name}")
    except Exception as exc:
        print(f"  [harbor] WARNING: orphan container cleanup failed: {exc}")


def _find_result_json(job_dir):
    """Find the trial-level result.json (has verifier_result.rewards.reward)."""
    if not job_dir.exists():
        return None
    # Collect all result.json files, prefer trial-level (deeper paths)
    candidates = sorted(job_dir.rglob("result.json"),
                        key=lambda p: len(p.parts), reverse=True)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Trial-level result.json has verifier_result with rewards
            if "verifier_result" in data:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    # Fallback: return the first result.json found
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
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
    parser.add_argument("--timeout", type=int, default=18000,
                        help="Safety-net timeout in seconds (real per-task timeout is enforced by Harbor via task.toml)")
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
            # Trial-level result.json: verifier_result.rewards.reward
            # 注意 rewards 可能为 null（errored trial），需判空
            vres = result.get("verifier_result") or {}
            rewards = vres.get("rewards") or {}
            reward = rewards.get("reward",
                     result.get("reward",
                     result.get("success", 0)))
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
