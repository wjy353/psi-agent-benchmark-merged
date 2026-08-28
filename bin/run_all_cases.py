#!/usr/bin/env python3
"""Terminal-Bench 评测主流程 — 遍历 case，运行 agent + verifier，收集结果。

支持自由选择 case：
  python3 run_all_cases.py                          # 跑所有 enabled 的 case
  python3 run_all_cases.py --cases fix-git,caffe-cifar-10  # 只跑指定 case
  python3 run_all_cases.py --versions 2.1           # 只跑 2.1 版本
  python3 run_all_cases.py --difficulties 易,中     # 只跑易+中
  python3 run_all_cases.py --exclude "2.1/fix-git"  # 排除指定 case
  python3 run_all_cases.py --limit 5                # 只跑前 5 个
  python3 run_all_cases.py --list                    # 列出将运行的 case，不执行
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# 将 src/ 加入 Python 路径，以便在 psi-agent 目录中也能导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.container import (
    run_cmd, image_exists, ensure_task_downloaded, parse_task_toml,
    start_env_container, build_verifier_image, run_agent, run_verifier,
    cleanup_container,
)
# 注意：case_source 仅在使用 --refresh 时需要，改为惰性导入，
# 避免在没有部署 src/case_source.py 的环境（如仅运行 --cases）下因 ImportError 启动失败。

# ── 全局路径 ───────────────────────────────────────────────────────────────
WORKDIR = Path(os.environ.get(
    "TB_BENCH_WORKDIR",
    f"{os.environ.get('HOME', '/root')}/psi-agent-benchmark",
))
TASKS_DIR = WORKDIR / "tasks"
RESULTS_DIR = WORKDIR / "pilot_results"
MANIFEST_DIR = WORKDIR / "config"
MANIFEST_JSON = MANIFEST_DIR / "benchmark_manifest.json"
MANIFEST_MD = MANIFEST_DIR / "benchmark_manifest.md"
CONFIG_PATH = WORKDIR / "config" / "benchmark.yaml"
PSI_DIR = WORKDIR / "psi-agent"
HARBOR_BIN = os.environ.get("TB_HARBOR_BIN", "harbor")
UV_BIN = os.environ.get("TB_UV_BIN", "uv")
WORKSPACE = "examples/terminal_bench"
# Path to the terminal_bench workspace *source* directory on the host (before it is
# copied into psi-agent/examples/ by setup.sh). We resolve it relative to the
# benchmark project root so `docker run -v` can mount it into the container as
# /opt/psi-agent/workspace.
WORKSPACE_DIR = WORKDIR / "workspaces" / "terminal_bench"


# ── 日志 ───────────────────────────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "benchmark.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── 配置加载 ───────────────────────────────────────────────────────────────
def load_env():
    """从 .env 文件加载环境变量。"""
    for env_file in (WORKDIR / ".env", PSI_DIR / ".env"):
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    os.environ[key] = val
            return


def load_config():
    """从 benchmark.yaml 读取运行时配置（含 case_filter）。"""
    if not CONFIG_PATH.exists() or yaml is None:
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log(f"WARN: failed to parse {CONFIG_PATH}: {e}")
        return {}


def load_cases(
    *,
    cases_override=None,
    versions=None,
    difficulties=None,
    exclude=None,
    limit=None,
):
    """从 case_metadata.json 读取 case 列表，并应用筛选。

    两种模式：
      * 默认（无任何筛选参数）：仅返回 enabled=true 的 case（最初的精选 30 个）。
      * 显式筛选（cases_override / versions / difficulties / exclude 任一非空）：
        从全量 case（当前 163 个）中筛选，忽略 enabled 标记，
        即用户可从官网拉取的全部 case 里自由选择。

    benchmark.yaml 的 case_filter 作为基线：CLI 参数非 None 时覆盖同名字段。

    Args:
        cases_override: 指定 case 名称列表（如 ["fix-git", "caffe-cifar-10"]），
                        精确匹配全量池，忽略 version/difficulty 与 enabled 筛选。
        versions: 版本筛选列表（如 ["2.1"]），空/None = 不限。
        difficulties: 难度筛选列表（如 ["易", "中"]），空/None = 不限。
        exclude: 排除的 case key 列表（如 ["2.1/fix-git"]）。
        limit: 最多返回的 case 数量，None = 不限。

    Returns:
        筛选后的 case dict 列表。
    """
    meta_path = MANIFEST_DIR / "case_metadata.json"
    if not meta_path.exists():
        log(f"ERROR: case_metadata.json not found at {meta_path}")
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 读取 benchmark.yaml 的 case_filter 作为基线
    config = load_config()
    yaml_filter = config.get("case_filter", {})
    base_versions = set(yaml_filter.get("include_versions") or [])
    base_difficulties = set(yaml_filter.get("include_difficulties") or [])
    base_exclude = set(yaml_filter.get("exclude_cases") or [])

    # CLI 参数覆盖 yaml 配置（非 None 时覆盖）
    versions_set = set(versions) if versions is not None else base_versions
    difficulties_set = set(difficulties) if difficulties is not None else base_difficulties
    exclude_set = set(exclude) if exclude is not None else base_exclude

    # 如果 cases_override 指定了，直接按名称精确匹配
    if cases_override:
        cases_override_lower = {c.lower() for c in cases_override}
        result = []
        matched_lower = set()
        for key, info in sorted(metadata.items()):
            name = info.get("name", "")
            # 支持 "version/name" 完整 key 或纯 name 匹配
            if name.lower() in cases_override_lower:
                matched_lower.add(name.lower())
            if key.lower() in cases_override_lower:
                matched_lower.add(key.lower())
            if name.lower() in cases_override_lower or key.lower() in cases_override_lower:
                result.append({
                    "version": info["version"],
                    "name": info["name"],
                    "difficulty": info.get("difficulty", ""),
                    "domain": info.get("domain", ""),
                })
        # 提示未匹配的名称（常见原因：本地与服务器 case_metadata.json 不同步）
        unmatched = [c for c in cases_override if c.lower() not in matched_lower]
        if unmatched:
            log(f"WARN: {len(unmatched)} 个指定的 case 未在 case_metadata.json 中找到: {unmatched}")
            log("      若刚更新过本地元数据，请在服务器上 git pull 后重跑 setup.sh")
        if not result:
            log(f"ERROR: no matching cases found for: {cases_override}")
            log(f"  available keys: {list(metadata.keys())}")
        if limit is not None:
            result = result[:limit]
        return result

    # 正常筛选流程
    # 是否处于“显式筛选”模式：
    #   - 有任一筛选参数（version / difficulty / exclude 非空）→ 从全量 163 个 case 中筛选，
    #     忽略 enabled（让用户能从全部 case 里自由选择）
    #   - 无任何筛选参数 → 仅返回 enabled=true 的默认 30 个 case（最初的精选集）
    has_filter = bool(versions_set) or bool(difficulties_set) or bool(exclude_set)

    result = []
    for key, info in sorted(metadata.items()):
        # 默认模式：只跑 enabled 的 case（即最初的 30 个精选）
        if not has_filter and not info.get("enabled", True):
            continue
        version = info["version"]
        name = info["name"]
        difficulty = info.get("difficulty", "")

        # 版本筛选
        if versions_set and version not in versions_set:
            continue
        # 难度筛选
        if difficulties_set and difficulty not in difficulties_set:
            continue
        # 排除列表
        if key in exclude_set:
            continue

        result.append({
            "version": version,
            "name": name,
            "difficulty": difficulty,
            "domain": info.get("domain", ""),
        })

    if limit is not None:
        result = result[:limit]
    return result


def load_manifest():
    if MANIFEST_JSON.exists():
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _write_manifest_md(manifest)


def _write_manifest_md(manifest):
    rows = [(item.get("order", 0), item) for key, item in manifest.items() if key != "_meta"]
    rows.sort(key=lambda x: x[0])
    lines = ["# TB Benchmark 结果清单\n\n"]
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("| # | 版本 | 任务名 | 领域 | 难度 | 状态 | Reward | 耗时 | 备注 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    total_reward = 0
    completed = 0
    for _, item in rows:
        reward = item.get("reward", "")
        if reward not in ("", None, "unknown"):
            completed += 1
            try:
                total_reward += float(reward)
            except Exception:
                pass
        lines.append(
            f"| {item.get('order', '')} | {item['version']} | {item['name']} | "
            f"{item.get('domain', '')} | {item.get('difficulty', '')} | "
            f"{item.get('agent_status', '')} | {reward} | "
            f"{item.get('elapsed_sec', '')} | {item.get('note', '')} |\n"
        )
    lines.append(f"\n汇总：完成 {completed}/{len(rows)} 个 case，总 reward {total_reward:.2f}。\n")
    with open(MANIFEST_MD, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def get_agent_version():
    ref = os.environ.get("PSI_AGENT_REF", "main")
    try:
        commit = __import__("subprocess").check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PSI_DIR, text=True
        ).strip()
        return f"{ref}@{commit}"
    except Exception:
        return ref


# ── 交互式选择 ─────────────────────────────────────────────────────────────
def pick_cases_interactively():
    """交互式选择 case：按版本分组展示，输入编号选择。

    支持的输入格式：
      1,3,5          — 选择编号 1、3、5
      2-6            — 选择编号 2 到 6（含边界）
      1,3-5,10       — 混合使用
      a              — 选择全部
      2.1            — 选择 TB 2.1 全部
      3.0            — 选择 TB 3.0 全部
      q              — 放弃并退出

    Returns:
        选中的 case dict 列表；放弃时返回 None。
    """
    meta_path = MANIFEST_DIR / "case_metadata.json"
    if not meta_path.exists():
        print(f"ERROR: case_metadata.json not found at {meta_path}")
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 按 version 分组，组内按名称排序
    by_version = {}
    for key, info in sorted(metadata.items()):
        by_version.setdefault(info["version"], []).append((key, info))

    # 展示编号清单
    numbered = []  # [(编号, key, info)]
    print()
    for version in sorted(by_version.keys()):
        cases = by_version[version]
        print(f"─── TB {version}（{len(cases)} 个）───")
        for key, info in cases:
            idx = len(numbered) + 1
            numbered.append((idx, key, info))
            flag = "" if info.get("enabled", True) else "  [已禁用]"
            print(f"  {idx:>2}. {info['name']:<40} {info.get('difficulty', ''):<4} {info.get('domain', '')}{flag}")
        print()

    print("输入编号选择（示例: 1,3-5,12  或  2.1  或  3.0  或  a 全选，q 放弃）:")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已放弃选择")
            return None
        if not raw:
            continue
        if raw.lower() == "q":
            print("已放弃选择")
            return None
        if raw.lower() == "a":
            selected_keys = {k for _, k, _ in numbered}
        elif raw in by_version:
            selected_keys = {k for k, _ in by_version[raw]}
        else:
            selected_keys = set()
            ok = True
            for part in raw.replace("，", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo, _, hi = part.partition("-")
                    try:
                        lo_i, hi_i = int(lo), int(hi)
                    except ValueError:
                        print(f"  无效范围: {part}")
                        ok = False
                        break
                    if lo_i < 1 or hi_i > len(numbered) or lo_i > hi_i:
                        print(f"  编号超出范围 1-{len(numbered)}: {part}")
                        ok = False
                        break
                    for n in range(lo_i, hi_i + 1):
                        selected_keys.add(numbered[n - 1][1])
                else:
                    try:
                        n = int(part)
                    except ValueError:
                        print(f"  无效编号: {part}")
                        ok = False
                        break
                    if n < 1 or n > len(numbered):
                        print(f"  编号超出范围 1-{len(numbered)}: {n}")
                        ok = False
                        break
                    selected_keys.add(numbered[n - 1][1])
            if not ok or not selected_keys:
                continue

        # 按清单顺序输出选中的 case
        result = [
            {
                "version": info["version"],
                "name": info["name"],
                "difficulty": info.get("difficulty", ""),
                "domain": info.get("domain", ""),
            }
            for _, key, info in numbered
            if key in selected_keys
        ]
        # 确认
        print(f"\n已选择 {len(result)} 个 case:")
        for c in result:
            print(f"  {c['version']}/{c['name']}")
        try:
            confirm = input("确认执行? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已放弃选择")
            return None
        if confirm in ("", "y", "yes"):
            return result
        print()


# ── 主流程 ─────────────────────────────────────────────────────────────────
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Terminal-Bench 评测主控 — 支持 case 自由选择",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 跑所有 enabled 的 case（默认行为）
  python3 run_all_cases.py

  # 只跑指定 case（按名称匹配，忽略版本/难度筛选）
  python3 run_all_cases.py --cases fix-git,caffe-cifar-10
  python3 run_all_cases.py --cases "2.1/fix-git" --cases "3.0/music-harmony"

  # 按版本/难度筛选
  python3 run_all_cases.py --versions 2.1
  python3 run_all_cases.py --versions 2.1,3.0 --difficulties 易
  python3 run_all_cases.py --difficulties 易,中

  # 排除指定 case
  python3 run_all_cases.py --exclude "2.1/fix-git" --exclude "3.0/gpt2-codegolf"

  # 限制数量（只跑前 N 个）
  python3 run_all_cases.py --limit 5

  # 交互式选择：按 TB 2.1 / 3.0 分组列出，输入编号挑选
  python3 run_all_cases.py --pick

  # 列出将运行的 case（不执行评测）
  python3 run_all_cases.py --list
  python3 run_all_cases.py --versions 3.0 --difficulties 难 --list

  # 组合使用：只跑 3.0 容易的，最多 3 个
  python3 run_all_cases.py --versions 3.0 --difficulties 易 --limit 3
""",
    )
    parser.add_argument(
        "--cases", "-c", action="append", default=None, metavar="NAME[,NAME...]",
        help="精确指定 case 名称（逗号分隔），可多次使用。忽略版本/难度筛选",
    )
    parser.add_argument(
        "--versions", "-v", default=None, metavar="V[,V...]",
        help="版本筛选，逗号分隔（如 2.1 或 2.1,3.0）",
    )
    parser.add_argument(
        "--difficulties", "-d", default=None, metavar="D[,D...]",
        help="难度筛选，逗号分隔（如 易 或 易,中,难）",
    )
    parser.add_argument(
        "--exclude", action="append", default=None, metavar="KEY",
        help="排除的 case key（如 2.1/fix-git），可多次使用",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None, metavar="N",
        help="最多运行的 case 数量",
    )
    parser.add_argument(
        "--pick", "-p", action="store_true",
        help="交互式选择 case：按 TB 2.1/3.0 分组展示，输入编号挑选",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="运行前先从官网拉取 TB 2.1/3.0 全量 case，刷新候选池",
    )
    parser.add_argument(
        "--refresh-meta", action="store_true",
        help="配合 --refresh，同时抓取 task.toml 的 difficulty/domain（需 GITHUB_TOKEN）",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出将运行的 case 后退出，不执行评测",
    )
    return parser.parse_args()


def _split_csv(s):
    """将逗号分隔的字符串拆为列表，None 返回 None。"""
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


# ── 主流程 ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 运行前从官网刷新候选池（--refresh）
    if args.refresh:
        from src.case_source import refresh_case_metadata
        meta_path = MANIFEST_DIR / "case_metadata.json"
        token = os.environ.get("GITHUB_TOKEN")
        try:
            n = refresh_case_metadata(
                meta_path, token=token, with_meta=args.refresh_meta,
                log=lambda m: log(m),
            )
            log(f"已从官网刷新候选池：共 {n} 个 case（写入 {meta_path}）")
        except Exception as e:
            log(f"WARN: 从官网刷新候选池失败，将使用现有 case_metadata.json: {e}")

    # 交互式选择模式：--pick 优先于其他 case 参数
    picked_cases = None
    if args.pick:
        picked_cases = pick_cases_interactively()
        if picked_cases is None:
            print("未选择任何 case，退出")
            sys.exit(0)
        if not picked_cases:
            print("选择的 case 为空，退出")
            sys.exit(1)
        cases_override = [c["name"] for c in picked_cases]
        # 让 manifest 记录本轮为 pick 模式
        args.cases = [",".join(cases_override)]
    else:
        cases_override = None
        if args.cases:
            cases_override = []
            for chunk in args.cases:
                cases_override.extend(_split_csv(chunk))

    load_env()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    versions = _split_csv(args.versions)
    difficulties = _split_csv(args.difficulties)
    exclude = args.exclude if args.exclude else None

    cases = load_cases(
        cases_override=cases_override,
        versions=versions,
        difficulties=difficulties,
        exclude=exclude,
        limit=args.limit,
    )

    if not cases:
        log("ERROR: no cases matched the given filters")
        sys.exit(1)

    # --list 模式：打印 case 列表后退出
    if args.list:
        print(f"\n将运行 {len(cases)} 个 case：\n")
        print(f"{'#':>3}  {'版本':<6} {'难度':<5} {'领域':<28} {'名称'}")
        print(f"{'─'*3}  {'─'*6} {'─'*5} {'─'*28} {'─'*30}")
        for i, c in enumerate(cases, 1):
            print(f"{i:>3}  {c['version']:<6} {c['difficulty']:<5} {c['domain']:<28} {c['name']}")
        print(f"\n共 {len(cases)} 个 case。去掉 --list 以执行评测。")
        return

    manifest = load_manifest()
    model = os.environ.get("PSI_AI_MODEL", "unknown")
    agent_version = get_agent_version()
    start_epoch = time.time()
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    manifest["_meta"] = {
        "model": model,
        "agent_version": agent_version,
        "start_time": start_ts,
        "start_epoch": start_epoch,
        "total_cases": len(cases),
        "case_filter": {
            "cases": cases_override,
            "versions": versions,
            "difficulties": difficulties,
            "exclude": exclude,
            "limit": args.limit,
        },
    }
    save_manifest(manifest)
    log(f"Benchmark started: model={model}, agent={agent_version}, cases={len(cases)}")

    for idx, case in enumerate(cases, 1):
        version = case["version"]
        name = case["name"]
        key = f"{version}/{name}"
        image_tag = f"tb-{version}-{name}:latest"
        verifier_tag = f"tb-{version}-{name}-verifier:latest"
        container_name = f"tb-agent-{version}-{name}"
        result_dir = RESULTS_DIR / name
        result_dir.mkdir(parents=True, exist_ok=True)

        # 跳过已完成的 case
        if key in manifest and manifest[key].get("reward") not in ("", "unknown"):
            log(f"=== [{idx}/{len(cases)}] {key} already completed (reward={manifest[key].get('reward')}), skipping ===")
            continue

        log(f"=== [{idx}/{len(cases)}] {key} ===")
        start_time = time.time()
        item = {
            "order": idx,
            "version": version,
            "name": name,
            "domain": case.get("domain", ""),
            "difficulty": case.get("difficulty", ""),
            "image_tag": image_tag,
            "agent_status": "pending",
            "reward": "",
            "elapsed_sec": "",
            "note": "",
            "result_dir": str(result_dir),
        }

        try:
            # 1. 下载任务
            task_dir = ensure_task_downloaded(name, TASKS_DIR, HARBOR_BIN, log_fn=log)
            if task_dir is None:
                item["agent_status"] = "download_failed"
                item["note"] = "harbor download failed"
                manifest[key] = item
                save_manifest(manifest)
                continue

            task_toml = parse_task_toml(task_dir)
            agent_timeout = min(task_toml.get("agent", {}).get("timeout_sec", 1800), 3600)

            # 2. 检查镜像
            if not image_exists(image_tag):
                item["agent_status"] = "image_missing"
                item["note"] = f"image {image_tag} not found"
                manifest[key] = item
                save_manifest(manifest)
                continue

            # 3. 启动容器（挂载 psi-agent + workspace + task + results 进容器）
            if not start_env_container(
                container_name, image_tag,
                psi_dir=PSI_DIR,
                workspace_dir=WORKSPACE_DIR,
                task_dir=task_dir,
                result_dir=result_dir,
                log_fn=log,
            ):
                item["agent_status"] = "container_failed"
                item["note"] = "failed to start env container"
                manifest[key] = item
                save_manifest(manifest)
                continue

            # 4. 构建 verifier（如需要）
            verifier_mode = task_toml.get("verifier", {}).get("environment_mode", "same")
            if verifier_mode == "separate":
                if not build_verifier_image(task_dir, verifier_tag, log_fn=log):
                    item["agent_status"] = "verifier_build_failed"
                    item["note"] = "failed to build verifier image"
                    manifest[key] = item
                    save_manifest(manifest)
                    cleanup_container(container_name, log_fn=log)
                    continue

            # 5. 运行 agent
            rc, status = run_agent(
                container_name, task_dir, result_dir,
                PSI_DIR, UV_BIN, WORKSPACE, agent_timeout,
                log_fn=log,
            )
            item["agent_status"] = status if rc == 0 else f"agent_{status}_rc{rc}"

            # 6. 运行 verifier
            reward = run_verifier(container_name, task_dir, task_toml, verifier_tag, result_dir, log_fn=log)
            item["reward"] = reward

            # 7. 清理
            cleanup_container(container_name, log_fn=log)

            elapsed = int(time.time() - start_time)
            item["elapsed_sec"] = elapsed
            item["note"] = f"completed in {elapsed}s"
            manifest[key] = item
            save_manifest(manifest)

        except Exception as e:
            log(f"Exception for {key}: {e}\n{traceback.format_exc()}")
            item["agent_status"] = "error"
            item["note"] = str(e)[:200]
            cleanup_container(container_name, log_fn=log)
            manifest[key] = item
            save_manifest(manifest)

    # 汇总
    end_epoch = time.time()
    elapsed = int(end_epoch - start_epoch)
    meta = manifest.get("_meta", {})
    meta.update({
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_epoch": end_epoch,
        "elapsed_sec": elapsed,
    })
    manifest["_meta"] = meta
    log(f"=== Benchmark complete in {elapsed}s ===")
    save_manifest(manifest)


if __name__ == "__main__":
    main()