"""
Unified benchmark orchestrator.
Dispatches to the right runner based on --benchmark.
Each benchmark (tb-2.1, tb-3.0, tau2, gaia) generates its own separate report
with version-specific data format and fields.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = ["tb-2.1", "tb-3.0", "tau2", "gaia", "all"]

TB_RUNNER = REPO_ROOT / "run_all_cases.py"
TB_REPORTER = REPO_ROOT / "generate_report.py"
TAU2_GAIA_RUNNER = REPO_ROOT / "scripts" / "psi_agent_benchmark.py"

MANIFEST_PATHS = [
    REPO_ROOT / "config" / "benchmark_manifest.json",
    REPO_ROOT / "manifests" / "benchmark_manifest.json",
]

REPORT_DIRS = {
    "tb-2.1": REPO_ROOT / "reports" / "tb-2.1",
    "tb-3.0": REPO_ROOT / "reports" / "tb-3.0",
    "tau2": REPO_ROOT / "reports" / "tau2",
    "gaia": REPO_ROOT / "reports" / "gaia",
}

# -- Per-benchmark data schema (for documentation / future report generators) --

BENCHMARK_SCHEMAS = {
    "tb-2.1": {
        "verifier_mode": "same",
        "scoring": "binary (0/1) — artifact check in-container",
        "fields": [
            "case_key", "version", "difficulty", "domain",
            "reward", "turns", "tokens", "cost", "duration_s",
            "verifier_stdout", "verifier_stderr",
        ],
        "report_sections": [
            "Overall Score (pass_rate, avg_turns, avg_tokens, total_cost)",
            "By Difficulty (easy/medium/hard)",
            "By Domain (software/ml/database/engineering)",
            "Detailed Results (per-case: reward, turns, tokens, verifier output)",
            "Intermediate Results (session/agent/verifier logs)",
            "Error Analysis (9-class taxonomy: turn_limit/compilation/runtime/verifier_rejection/env/timeout/logic/agent_crash/unknown)",
        ],
    },
    "tb-3.0": {
        "verifier_mode": "separate",
        "scoring": "binary (0/1) — artifact check in separate container via docker cp + bind mount",
        "fields": [
            "case_key", "version", "difficulty", "domain",
            "reward", "turns", "tokens", "cost", "duration_s",
            "verifier_stdout", "verifier_stderr",
            "verifier_container_image", "overlay_image",
            "docker_cp_path", "bind_mount_path",
        ],
        "report_sections": [
            "Overall Score (pass_rate, avg_turns, avg_tokens, total_cost)",
            "By Difficulty (easy/medium/hard)",
            "By Domain (software/ml/database/engineering/formal)",
            "Detailed Results (per-case: reward, turns, tokens, verifier output, container isolation info)",
            "Intermediate Results (session/agent/verifier logs)",
            "Error Analysis (9-class + separate_verifier_failure + container_isolation_error)",
            "Container Isolation Audit (docker cp success, bind mount integrity, overlay image coverage)",
        ],
    },
    "tau2": {
        "verifier_mode": "reward + db_check + assertion",
        "scoring": "multi-dimensional (reward 0/1, db_check pass/fail, assertion pass/fail)",
        "fields": [
            "task_id", "domain", "subset",
            "reward", "db_check_pass", "assertion_pass",
            "user_messages", "tool_calls", "turns", "tokens", "duration_s",
        ],
        "report_sections": [
            "Overall Score (resolution_rate, avg_turns, avg_user_messages)",
            "By Domain (airline/retail/telecom/banking_knowledge/mock)",
            "Detailed Results (per-task: reward, db_check, assertion, tool stats)",
            "Failure Snapshots (conversation excerpt on failure)",
            "Tool Usage Stats (which tools called most, success rate)",
        ],
    },
    "gaia": {
        "verifier_mode": "gaia_scorer (exact match)",
        "scoring": "continuous (0.0-1.0) — exact string/number match against ground truth",
        "fields": [
            "task_id", "level", "subset", "task_type",
            "score", "is_correct",
            "file_count", "search_count", "turns", "tokens", "duration_s",
        ],
        "report_sections": [
            "Overall Score (avg_score, exact_match_rate, partial_match_rate)",
            "By Level (level 1/2/3)",
            "By Task Type (text/file/multimodal/reasoning)",
            "Detailed Results (per-task: score, match_type, file/search stats)",
            "Failure Analysis (wrong answer type: missing_file/wrong_number/incomplete_reasoning)",
        ],
    },
}


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    """Run a subprocess, stream output to terminal, return exit code."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return result.returncode


def _find_manifest() -> Path | None:
    """Find the most recently written benchmark manifest."""
    for p in MANIFEST_PATHS:
        if p.exists():
            return p
    return None


def _copy_manifest(dest_dir: Path) -> Path | None:
    """Copy manifest to report dir so reports are self-contained."""
    src = _find_manifest()
    if src is None:
        return None
    dest = dest_dir / "manifest.json"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


# ---- TB 2.1 ----

def run_tb_2_1(args: argparse.Namespace) -> int:
    """Run Terminal-Bench 2.1 cases only."""
    run_id = args.run_id or f"tb21-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  Terminal-Bench 2.1 | run_id={run_id}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(TB_RUNNER), "--versions", "2.1"]
    if args.cases:
        cmd += ["--cases", args.cases]
    if args.difficulties:
        cmd += ["--difficulties", args.difficulties]
    if args.exclude:
        for ex in args.exclude:
            cmd += ["--exclude", ex]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--list"]

    rc = _run(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"  [TB 2.1] runner exited with {rc}")
    return rc


def report_tb_2_1(args: argparse.Namespace) -> int:
    """Generate TB 2.1 report with 2.1-specific data format."""
    run_id = args.run_id or "latest"
    out_dir = REPORT_DIRS["tb-2.1"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "report.md"

    manifest = _copy_manifest(out_dir)
    cmd = [sys.executable, str(TB_REPORTER), "--out", str(out_file)]
    if manifest:
        cmd += ["--manifest", str(manifest)]

    rc = _run(cmd, cwd=REPO_ROOT)
    if rc == 0:
        print(f"  [TB 2.1] report: {out_file}")
    return rc


# ---- TB 3.0 ----

def run_tb_3_0(args: argparse.Namespace) -> int:
    """Run Terminal-Bench 3.0 cases only."""
    run_id = args.run_id or f"tb30-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  Terminal-Bench 3.0 | run_id={run_id}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(TB_RUNNER), "--versions", "3.0"]
    if args.cases:
        cmd += ["--cases", args.cases]
    if args.difficulties:
        cmd += ["--difficulties", args.difficulties]
    if args.exclude:
        for ex in args.exclude:
            cmd += ["--exclude", ex]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--list"]

    rc = _run(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"  [TB 3.0] runner exited with {rc}")
    return rc


def report_tb_3_0(args: argparse.Namespace) -> int:
    """Generate TB 3.0 report with 3.0-specific data format."""
    run_id = args.run_id or "latest"
    out_dir = REPORT_DIRS["tb-3.0"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "report.md"

    manifest = _copy_manifest(out_dir)
    cmd = [sys.executable, str(TB_REPORTER), "--out", str(out_file)]
    if manifest:
        cmd += ["--manifest", str(manifest)]

    rc = _run(cmd, cwd=REPO_ROOT)
    if rc == 0:
        print(f"  [TB 3.0] report: {out_file}")
    return rc


# ---- tau2 ----

def run_tau2(args: argparse.Namespace) -> int:
    """Run tau2-bench via local adapter."""
    run_id = args.run_id or f"tau2-{time.strftime('%Y%m%d-%H%M%S')}"
    subset = args.subset or "quick_3"
    print(f"\n{'='*60}")
    print(f"  tau2-bench | run_id={run_id} | subset={subset}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(TAU2_GAIA_RUNNER), "run",
        "--benchmark", "tau2",
        "--subset", subset,
        "--run-id", run_id,
    ]
    if args.psi_root:
        cmd += ["--psi-root", args.psi_root]
    if args.tau2_root:
        cmd += ["--tau2-root", args.tau2_root]
    if args.uv:
        cmd += ["--uv", args.uv]
    if args.api_key_file:
        cmd += ["--api-key-file", args.api_key_file]
    if args.dry_run:
        cmd += ["--dry-run"]

    rc = _run(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"  [tau2] runner exited with {rc}")
    return rc


def report_tau2(args: argparse.Namespace) -> int:
    """Generate tau2 report (by domain, multi-dimensional scoring)."""
    if not args.run_id:
        print("  [tau2] --run-id is required for report")
        return 1
    cmd = [
        sys.executable, str(TAU2_GAIA_RUNNER), "report",
        "--benchmark", "tau2",
        "--run-id", args.run_id,
    ]
    rc = _run(cmd, cwd=REPO_ROOT)
    if rc == 0:
        print(f"  [tau2] report generated for run_id={args.run_id}")
    return rc


# ---- GAIA ----

def run_gaia(args: argparse.Namespace) -> int:
    """Run GAIA via local execution."""
    run_id = args.run_id or f"gaia-{time.strftime('%Y%m%d-%H%M%S')}"
    subset = args.subset or "level1_smoke"
    print(f"\n{'='*60}")
    print(f"  GAIA | run_id={run_id} | subset={subset}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(TAU2_GAIA_RUNNER), "run",
        "--benchmark", "gaia",
        "--subset", subset,
        "--run-id", run_id,
    ]
    if args.psi_root:
        cmd += ["--psi-root", args.psi_root]
    if args.gaia_data_root:
        cmd += ["--gaia-data-root", args.gaia_data_root]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--dry-run"]

    rc = _run(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"  [GAIA] runner exited with {rc}")
    return rc


def report_gaia(args: argparse.Namespace) -> int:
    """Generate GAIA report (by level, continuous scoring)."""
    if not args.run_id:
        print("  [GAIA] --run-id is required for report")
        return 1
    cmd = [
        sys.executable, str(TAU2_GAIA_RUNNER), "report",
        "--benchmark", "gaia",
        "--run-id", args.run_id,
    ]
    rc = _run(cmd, cwd=REPO_ROOT)
    if rc == 0:
        print(f"  [GAIA] report generated for run_id={args.run_id}")
    return rc


# -- Dispatch tables --

RUNNERS = {
    "tb-2.1": run_tb_2_1,
    "tb-3.0": run_tb_3_0,
    "tau2": run_tau2,
    "gaia": run_gaia,
}

REPORTERS = {
    "tb-2.1": report_tb_2_1,
    "tb-3.0": report_tb_3_0,
    "tau2": report_tau2,
    "gaia": report_gaia,
}

ALL_BENCHMARKS = ["tb-2.1", "tb-3.0", "tau2", "gaia"]


def cmd_run(args: argparse.Namespace) -> int:
    """Execute benchmark(s) and generate separate reports for each."""
    benchmarks = ALL_BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    results: dict[str, dict] = {}
    overall_rc = 0
    base_run_id = args.run_id

    for bench in benchmarks:
        # Each benchmark gets its own run_id suffix when running all
        if args.benchmark == "all" and base_run_id:
            args.run_id = f"{base_run_id}-{bench}"
        else:
            args.run_id = base_run_id

        runner = RUNNERS[bench]
        rc = runner(args)
        results[bench] = {"run_rc": rc}
        if rc != 0:
            overall_rc = rc

        if not args.dry_run and not args.no_report:
            reporter = REPORTERS[bench]
            rep_rc = reporter(args)
            results[bench]["report_rc"] = rep_rc

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for bench, rcs in results.items():
        run_str = f"run={'OK' if rcs.get('run_rc', 1) == 0 else 'FAIL'}"
        rep_str = f"report={'OK' if rcs.get('report_rc', 1) == 0 else 'SKIP/FAIL'}"
        print(f"  {bench:20s} | {run_str} | {rep_str}")

    return overall_rc


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report(s) only."""
    benchmarks = ALL_BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    for bench in benchmarks:
        REPORTERS[bench](args)

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available cases/subsets for a benchmark."""
    if args.benchmark in ("tb-2.1", "tb-3.0"):
        version = "2.1" if args.benchmark == "tb-2.1" else "3.0"
        _run([sys.executable, str(TB_RUNNER), "--versions", version, "--list"], cwd=REPO_ROOT)
    elif args.benchmark == "tau2":
        _run([sys.executable, str(TAU2_GAIA_RUNNER), "list-subsets", "--benchmark", "tau2"], cwd=REPO_ROOT)
    elif args.benchmark == "gaia":
        _run([sys.executable, str(TAU2_GAIA_RUNNER), "list-subsets", "--benchmark", "gaia"], cwd=REPO_ROOT)
    elif args.benchmark == "all":
        for bench in ALL_BENCHMARKS:
            print(f"\n--- {bench} ---")
            cmd_list(argparse.Namespace(benchmark=bench))
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    """Print data schema for a benchmark."""
    if args.benchmark == "all":
        for bench in ALL_BENCHMARKS:
            print(f"\n--- {bench} ---")
            _print_schema(bench)
    else:
        _print_schema(args.benchmark)
    return 0


def _print_schema(bench: str) -> None:
    schema = BENCHMARK_SCHEMAS.get(bench, {})
    print(f"  verifier_mode: {schema.get('verifier_mode', 'N/A')}")
    print(f"  scoring:       {schema.get('scoring', 'N/A')}")
    print(f"  fields:")
    for f in schema.get("fields", []):
        print(f"    - {f}")
    print(f"  report_sections:")
    for s in schema.get("report_sections", []):
        print(f"    - {s}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psi-agent-benchmark",
        description="Unified benchmark runner: TB 2.1 + TB 3.0 + tau2 + GAIA",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Run benchmark(s)")
    p_run.add_argument("--benchmark", "-b", choices=BENCHMARKS, default="all",
                        help="Which benchmark to run (default: all = tb-2.1 + tb-3.0 + tau2 + gaia)")
    p_run.add_argument("--run-id", default="", help="Run identifier (auto-generated if empty; each bench gets suffix when -b all)")
    p_run.add_argument("--dry-run", action="store_true", help="Preview without executing")
    p_run.add_argument("--no-report", action="store_true", help="Skip report generation")

    # TB-specific (applies to both 2.1 and 3.0)
    p_run.add_argument("--cases", "-c", default="", help="TB: comma-separated case names")
    p_run.add_argument("--difficulties", "-d", default="", help="TB: difficulty filter")
    p_run.add_argument("--exclude", action="append", default=[], help="TB: exclude case key")
    p_run.add_argument("--limit", "-n", type=int, default=0, help="TB/GAIA: max cases")

    # tau2/GAIA shared
    p_run.add_argument("--subset", "-s", default="", help="tau2/GAIA: subset name (default: quick_3 / level1_smoke)")
    p_run.add_argument("--psi-root", default="", help="psi-agent repo root")
    p_run.add_argument("--tau2-root", default="", help="tau2-bench repo root")
    p_run.add_argument("--uv", default="", help="uv executable path")
    p_run.add_argument("--api-key-file", default="", help="API key file path")
    p_run.add_argument("--gaia-data-root", default="", help="GAIA data root directory")
    p_run.set_defaults(func=cmd_run)

    # -- report --
    p_rep = sub.add_parser("report", help="Generate report(s) only")
    p_rep.add_argument("--benchmark", "-b", choices=BENCHMARKS, default="all")
    p_rep.add_argument("--run-id", default="", help="Run identifier")
    p_rep.set_defaults(func=cmd_report)

    # -- list --
    p_list = sub.add_parser("list", help="List available cases/subsets")
    p_list.add_argument("--benchmark", "-b", choices=BENCHMARKS, default="all")
    p_list.set_defaults(func=cmd_list)

    # -- schema --
    p_schema = sub.add_parser("schema", help="Print data schema for a benchmark")
    p_schema.add_argument("--benchmark", "-b", choices=BENCHMARKS, default="all")
    p_schema.set_defaults(func=cmd_schema)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
