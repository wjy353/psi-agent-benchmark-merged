#!/usr/bin/env python3
"""从 Terminal-Bench 官网拉取 TB 2.1 / 3.0 全部 case，生成 case_metadata.json。

用法:
    # 仅拉取 case 名称（默认，约 2 次 API 调用）
    python fetch_cases.py

    # 同时抓取每个 task.toml 的 difficulty / domain（需要 GitHub token）
    GITHUB_TOKEN=ghp_xxx python fetch_cases.py --with-meta

    # 指定输出路径
    python fetch_cases.py --out config/case_metadata.json

    # 自定义 3.0 数据源（官方仓库发布后替换此处）
    python fetch_cases.py --3.0-repo owner/repo

生成的文件与 run_all_cases.py / --pick 选择菜单完全兼容（version/name 为键）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 把仓库根目录加入路径，以便导入 src.case_source
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.case_source import (
    DEFAULT_SOURCES,
    build_case_metadata,
    merge_existing,
    refresh_case_metadata,
)

DEFAULT_OUT = ROOT / "config" / "case_metadata.json"


def parse_repo(s):
    """owner/repo[:branch] -> dict"""
    owner, _, rest = s.partition("/")
    repo, _, branch = rest.partition(":")
    return {"owner": owner, "repo": repo, "branch": branch or None}


def main():
    parser = argparse.ArgumentParser(description="从官网拉取 TB 2.1/3.0 全部 case")
    parser.add_argument(
        "--out", "-o", default=str(DEFAULT_OUT),
        help=f"输出路径（默认 {DEFAULT_OUT}）",
    )
    parser.add_argument(
        "--with-meta", action="store_true",
        help="同时抓取 task.toml 的 difficulty/domain（需 GitHub token，否则易限流）",
    )
    parser.add_argument(
        "--token", default=os.getenv("GITHUB_TOKEN"),
        help="GitHub token（也可用环境变量 GITHUB_TOKEN），用于 --with-meta 提额",
    )
    parser.add_argument("--2.1-repo", dest="repo_2_1", default=None,
                        help="覆盖 2.1 数据源，格式 owner/repo[:branch]")
    parser.add_argument("--3.0-repo", dest="repo_3_0", default=None,
                        help="覆盖 3.0 数据源，格式 owner/repo[:branch]")
    args = parser.parse_args()

    sources = dict(DEFAULT_SOURCES)
    if args.repo_2_1:
        sources["2.1"] = parse_repo(args.repo_2_1)
    if args.repo_3_0:
        sources["3.0"] = parse_repo(args.repo_3_0)

    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARN: 读取现有 {out_path} 失败，将覆盖: {e}")

    print(f"==> 从官网拉取 TB 2.1 / 3.0 全部 case ...")
    fresh = build_case_metadata(token=args.token, with_meta=args.with_meta, sources=sources)
    fresh = merge_existing(fresh, existing)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    by_version = {}
    with_meta = 0
    for entry in fresh.values():
        by_version.setdefault(entry["version"], 0)
        by_version[entry["version"]] += 1
        if entry.get("difficulty"):
            with_meta += 1
    print(f"==> 已写入 {out_path}")
    print(f"    总计 {len(fresh)} 个 case：" + ", ".join(
        f"TB {v}: {n}" for v, n in sorted(by_version.items())
    ))
    print(f"    含 difficulty/domain 元数据：{with_meta} 个" +
          ("" if args.with_meta else "（运行 --with-meta 可补全）"))


if __name__ == "__main__":
    main()
