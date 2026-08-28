#!/usr/bin/env python3
"""Generate a benchmark report from manifest + logs with detailed error analysis.

Sections:
  一、综合打分
  二、按版本统计
  三、按难度统计
  四、按领域统计
  五、详细结果
  六、Case 运行中间结果
  七、错误分析（新增）
    - 7.1 失败原因分类总览
    - 7.2 各 case 错误详情
    - 7.3 工具调用统计
    - 7.4 Skills/Tools 优化建议
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

WORKDIR = Path(os.environ.get("TB_BENCH_WORKDIR", f"{os.environ.get('HOME', '/root')}/psi-agent-benchmark"))
MANIFEST_JSON = WORKDIR / "config" / "benchmark_manifest.json"
METADATA_JSON = WORKDIR / "config" / "case_metadata.json"
RESULTS_DIR = WORKDIR / "pilot_results"
CHARS_PER_TOKEN = 4.0
LOG_TAIL_LINES = 30


# ═══════════════════════════════════════════════════════════════════════════
# 通用工具函数
# ═══════════════════════════════════════════════════════════════════════════

def read_text(path, default=""):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return default


def count_chars(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def tail_lines(text, n):
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""


USAGE_RE = re.compile(
    r"Request completed successfully \| usage prompt_tokens=(\d+) completion_tokens=(\d+) total_tokens=(\d+)"
)


def parse_real_usage(case_name):
    """从 ai.log 读取 psi-agent 记录的每次请求真实 usage。

    仅当 ai/server.py 已打补丁（成功行带 '| usage ...'）时才有数据；否则返回 None。
    """
    ai_log = RESULTS_DIR / case_name / "ai.log"
    text = read_text(ai_log, "")
    inp = out = tot = 0
    n = 0
    for m in USAGE_RE.finditer(text):
        inp += int(m.group(1))
        out += int(m.group(2))
        tot += int(m.group(3))
        n += 1
    if n == 0:
        return None
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot, "usage_requests": n}


def estimate_tokens(case_name):
    result_dir = RESULTS_DIR / case_name
    ai_log = result_dir / "ai.log"
    session_log = result_dir / "session.log"
    agent_out = result_dir / "agent_output.log"

    ai_text = read_text(ai_log, "")
    requests = ai_text.count("Request completed successfully")
    session_chars = count_chars(session_log)
    output_chars = count_chars(agent_out)

    input_tokens = int(session_chars * requests / 2 / CHARS_PER_TOKEN) if requests else 0
    output_tokens = int(output_chars / CHARS_PER_TOKEN)
    total_tokens = input_tokens + output_tokens
    token_source = "估算"

    # 优先用 ai/server.py 记录的真实 usage（每次请求的 prompt/completion/total）
    real = parse_real_usage(case_name)
    if real is not None:
        input_tokens = real["input_tokens"]
        output_tokens = real["output_tokens"]
        total_tokens = real["total_tokens"]
        token_source = "实测"

    return {
        "requests": requests,
        "session_chars": session_chars,
        "output_chars": output_chars,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "token_source": token_source,
    }


def reward_value(reward):
    if reward in ("", None, "unknown"):
        return None
    try:
        return float(reward)
    except Exception:
        return None


def is_pass(reward):
    return reward_value(reward) == 1.0


def is_fail(reward):
    val = reward_value(reward)
    return val is not None and val != 1.0


def is_unknown(reward):
    return reward in ("", None, "unknown") or reward_value(reward) is None


def format_elapsed(seconds):
    try:
        seconds = int(seconds)
    except Exception:
        return str(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ═══════════════════════════════════════════════════════════════════════════
# 错误分析模块
# ═══════════════════════════════════════════════════════════════════════════

# ── 错误类别定义 ──────────────────────────────────────────────────────────

ERROR_CATEGORIES = {
    "turn_limit": {
        "label": "轮次耗尽",
        "description": "Agent 达到最大工具调用轮次（默认 128），未能完成任务",
        "optimization_area": "Agent 策略 / 规划能力",
    },
    "compilation_error": {
        "label": "编译错误",
        "description": "Agent 生成的代码存在编译/构建错误（C/Python 语法、缺少依赖等）",
        "optimization_area": "代码生成 / 语言技能",
    },
    "runtime_error": {
        "label": "运行时错误",
        "description": "Agent 生成的代码运行时崩溃（TypeError, ValueError, KeyError, segfault 等）",
        "optimization_area": "代码调试 / 运行时验证",
    },
    "verifier_rejection": {
        "label": "Verifier 拒绝",
        "description": "Agent 自认为完成但 verifier 检查不通过（输出格式/内容不符合要求）",
        "optimization_area": "任务理解 / 输出验证",
    },
    "environment_error": {
        "label": "环境错误",
        "description": "容器启动失败、镜像缺失、下载失败等基础设施问题",
        "optimization_area": "基础设施 / 镜像管理",
    },
    "timeout": {
        "label": "超时",
        "description": "Agent 在规定时间内未完成（未达轮次上限但耗尽时间）",
        "optimization_area": "效率优化 / 策略精简",
    },
    "logic_error": {
        "label": "逻辑错误",
        "description": "代码可运行但计算结果错误（算法实现与规格不符）",
        "optimization_area": "推理能力 / 规格理解",
    },
    "agent_crash": {
        "label": "Agent 崩溃",
        "description": "Agent 进程异常退出（RC != 0，非轮次耗尽）",
        "optimization_area": "Agent 稳定性 / 错误恢复",
    },
    "unknown_failure": {
        "label": "未知失败",
        "description": "无法从日志中确定具体失败原因",
        "optimization_area": "日志增强 / 可观测性",
    },
}


# ── 日志解析正则 ──────────────────────────────────────────────────────────

# 工具调用匹配：Executing tool: 'bash' / 'read' / 'write' 等
TOOL_CALL_RE = re.compile(r"Executing tool: '(\w+)'")
# 工具结果错误指示
TOOL_ERROR_PATTERNS = [
    (re.compile(r"\[Exit code: (\d+)\]"), "exit_code"),
    (re.compile(r"Traceback \(most recent call last\)"), "python_traceback"),
    (re.compile(r"(?:error|Error|ERROR):\s+.+"), "error_message"),
]
# Python 异常类型
PYTHON_EXCEPTION_RE = re.compile(r"^(\w+Error|\w+Exception):", re.MULTILINE)
# 编译错误
COMPILE_ERROR_RES = [
    re.compile(r"gcc|g\+\+|clang|make|cmake.*error", re.IGNORECASE),
    re.compile(r"implicit declaration|undefined reference|cannot find", re.IGNORECASE),
    re.compile(r"SyntaxError|IndentationError|ImportError|ModuleNotFoundError"),
]
# 轮次上限
TURN_LIMIT_RES = [
    re.compile(r"Reached max tool rounds"),
    re.compile(r"stop_cause=agent_turn_limit"),
    re.compile(r"Session request incomplete.*agent_turn_limit"),
]
# Verifier 结果
VERIFIER_REWARD_RE = re.compile(r"^reward:\s*(\S+)", re.MULTILINE)
VERIFIER_RC_RE = re.compile(r"^verifier RC:\s*(\d+)", re.MULTILINE)
# Agent 状态异常
AGENT_CRASH_RES = [
    re.compile(r"agent_finished_rc\d+"),
    re.compile(r"agent_\w+_rc\d+"),
    re.compile(r"container_failed|download_failed|image_missing|verifier_build_failed"),
]
# 环境错误
ENV_ERROR_RES = [
    re.compile(r"download_failed|image_missing|container_failed|verifier_build_failed"),
]


def classify_error(case):
    """对单个 case 进行错误分类，返回主错误类别和详细信息。

    Args:
        case: 包含 name, agent_status, reward, result_dir 等字段的字典

    Returns:
        dict: {
            "category": 错误类别 key,
            "label": 类别中文标签,
            "detail": 具体错误信息（1-2 行）,
            "evidence": 日志中的证据片段,
            "tool_stats": {"bash": {calls, errors}, "read": {...}, ...},
        }
    """
    result_dir = Path(case["result_dir"])
    session_text = read_text(result_dir / "session.log", "")
    agent_text = read_text(result_dir / "agent_output.log", "")
    verifier_text = read_text(result_dir / "verifier.log", "")
    status = case.get("agent_status", "")
    reward = case.get("reward", "")

    # ── 工具调用统计 ──
    tool_stats = defaultdict(lambda: {"calls": 0, "errors": 0, "error_examples": []})
    for m in TOOL_CALL_RE.finditer(session_text):
        tool_name = m.group(1)
        tool_stats[tool_name]["calls"] += 1

    # 统计工具执行出错次数（以 Exit code != 0 和 Traceback 为指示）
    tool_error_re = re.compile(
        r"Tool result \('(\w+)'\):.*?(?:\[Exit code: [^0]\]|Traceback)"
    )
    for m in tool_error_re.finditer(session_text, re.DOTALL):
        tool_name = m.group(1)
        tool_stats[tool_name]["errors"] += 1
        snippet = m.group(0)[:200]
        if len(tool_stats[tool_name]["error_examples"]) < 3:
            tool_stats[tool_name]["error_examples"].append(snippet)

    # ── 按优先级逐项判断 ──

    # 1. 环境错误
    for er in ENV_ERROR_RES:
        if er.search(status):
            return _make_result("environment_error", status,
                               f"agent_status={status}", tool_stats)

    # 2. 轮次耗尽
    for tr in TURN_LIMIT_RES:
        if tr.search(session_text):
            turn_count = case.get("requests", 0)
            return _make_result("turn_limit",
                               f"达到最大轮次 ({turn_count} 次 API 调用)",
                               session_text[max(0, session_text.rfind("Reached max")):][:300],
                               tool_stats)

    # 3. Agent 崩溃（RC != 0 且非轮次耗尽）
    rc_match = re.search(r"rc(\d+)", status)
    if rc_match and int(rc_match.group(1)) != 0:
        if "agent_finished" in status:
            # Agent 完成了但返回非零退出码
            if session_text and not any(tr.search(session_text) for tr in TURN_LIMIT_RES):
                return _make_result("agent_crash",
                                   f"Agent 退出码非零 (rc={rc_match.group(1)})",
                                   session_text[-500:][:300],
                                   tool_stats)

    # 4. 对日志内容进行深度分析
    # 提取 Python 异常
    python_exceptions = PYTHON_EXCEPTION_RE.findall(session_text)
    # 提取编译错误
    has_compile_error = any(cr.search(session_text) for cr in COMPILE_ERROR_RES)

    if has_compile_error:
        # 找到具体编译错误行
        compile_lines = []
        for line in session_text.splitlines():
            if any(cr.search(line) for cr in COMPILE_ERROR_RES):
                compile_lines.append(line.strip()[:150])
                if len(compile_lines) >= 3:
                    break
        return _make_result("compilation_error",
                           "编译/构建失败",
                           "\n".join(compile_lines),
                           tool_stats)

    if python_exceptions:
        # 统计最常见的异常类型
        exc_counter = Counter(python_exceptions)
        top_exc = exc_counter.most_common(3)
        exc_summary = ", ".join(f"{e}({c}次)" for e, c in top_exc)
        # 找到第一个 traceback 的位置作为证据
        tb_pos = session_text.find("Traceback")
        evidence = session_text[tb_pos:tb_pos + 400] if tb_pos >= 0 else ""
        return _make_result("runtime_error",
                           f"运行时异常: {exc_summary}",
                           evidence,
                           tool_stats)

    # 5. Verifier 拒绝（Agent 完成了但 reward != 1）
    if status in ("finished", "") and is_fail(reward):
        verifier_rc_m = VERIFIER_RC_RE.search(verifier_text)
        verifier_rc = int(verifier_rc_m.group(1)) if verifier_rc_m else "?"
        # 判断是否是逻辑错误（verifier RC=0 但 reward != 1 通常意味着输出内容不对）
        if verifier_rc == 0 and reward_value(reward) == 0:
            # 区分"逻辑错误"和"格式不对"
            # 检查 agent_output 是否包含类似 "ALL CHECKS PASSED" 或 "PASS" 的声明
            if re.search(r"(?:ALL.*(?:CHECK|TEST).*(?:PASSED|PASS)|task is complete|Summary of.*work)",
                        agent_text, re.IGNORECASE):
                return _make_result("logic_error",
                                   f"Agent 自认完成但 verifier 判 reward={reward}（逻辑/精度不达标）",
                                   verifier_text[:300],
                                   tool_stats)
            else:
                return _make_result("verifier_rejection",
                                   f"Verifier 判 reward={reward}（输出未通过检查）",
                                   verifier_text[:300],
                                   tool_stats)
        else:
            return _make_result("verifier_rejection",
                               f"Verifier RC={verifier_rc}, reward={reward}",
                               verifier_text[:300],
                               tool_stats)

    # 6. 超时（未完成但也没有明确错误）
    if is_unknown(reward) and status not in ("", "pending"):
        elapsed = case.get("elapsed_sec", 0) or 0
        if elapsed > 0:
            return _make_result("timeout",
                               f"运行超时 ({format_elapsed(elapsed)})",
                               session_text[-300:],
                               tool_stats)

    # 7. 兜底
    return _make_result("unknown_failure",
                       f"未明确分类 (status={status}, reward={reward})",
                       session_text[-300:],
                       tool_stats)


def _make_result(category, detail, evidence, tool_stats):
    cat_info = ERROR_CATEGORIES.get(category, ERROR_CATEGORIES["unknown_failure"])
    return {
        "category": category,
        "label": cat_info["label"],
        "optimization_area": cat_info["optimization_area"],
        "detail": detail,
        "evidence": evidence[:500] if evidence else "",
        "tool_stats": dict(tool_stats),
    }


# ── 报告渲染 ──────────────────────────────────────────────────────────────

def build_error_analysis_md(cases):
    """为所有失败/unknown 的 case 生成错误分析章节。

    Returns:
        list[str]: markdown 行列表
    """
    lines = []
    lines.append("\n## 七、错误分析\n")
    lines.append(
        "以下对 reward != 1 的 case 逐个分析失败原因，提取工具调用统计，"
        "并给出 **通用 skills/tools 优化方向**，作为迭代改进的依据。\n"
    )

    # 只分析失败和 unknown 的 case
    failed_cases = [c for c in cases if not is_pass(c["reward"])]

    if not failed_cases:
        lines.append("> 本轮所有 case 均通过，无错误分析。\n")
        return lines

    # ── 7.1 失败原因分类总览 ──
    lines.append("\n### 7.1 失败原因分类总览\n")

    # 分析每个失败 case
    analyses = []
    for c in failed_cases:
        analysis = classify_error(c)
        analysis["case"] = c
        analyses.append(analysis)

    # 按类别统计
    cat_counter = Counter(a["category"] for a in analyses)
    opt_counter = Counter(a["optimization_area"] for a in analyses)

    lines.append("| 错误类别 | 数量 | 占比 | 优化方向 |")
    lines.append("| -------- | ---:| ---:| -------- |")
    for cat_key, count in cat_counter.most_common():
        cat_info = ERROR_CATEGORIES.get(cat_key, ERROR_CATEGORIES["unknown_failure"])
        pct = count / len(failed_cases) * 100
        lines.append(f"| {cat_info['label']} | {count} | {pct:.0f}% | {cat_info['optimization_area']} |")
    lines.append("\n")

    # 优化方向聚合
    lines.append("**优化方向按频次排序：**\n")
    for opt, count in opt_counter.most_common():
        related = [a for a in analyses if a["optimization_area"] == opt]
        names = [f"{a['case']['version']}/{a['case']['name']}" for a in related]
        lines.append(f"- **{opt}** ({count} 个 case): {', '.join(names)}\n")
    lines.append("\n")

    # ── 7.2 各 case 错误详情 ──
    lines.append("\n### 7.2 各 Case 错误详情\n")

    for a in analyses:
        c = a["case"]
        lines.append(f"\n#### {c['version']}/{c['name']}\n")
        lines.append(f"- **错误类别**: {a['label']}\n")
        lines.append(f"- **具体原因**: {a['detail']}\n")
        lines.append(f"- **优化方向**: {a['optimization_area']}\n")

        # 工具调用统计
        tool_stats = a.get("tool_stats", {})
        if tool_stats:
            lines.append("\n| 工具 | 调用次数 | 出错次数 | 错误率 |")
            lines.append("| ---- | -------:| -------:| ------:|")
            for tool, stats in sorted(tool_stats.items(), key=lambda x: -x[1]["calls"]):
                calls = stats["calls"]
                errors = stats["errors"]
                err_rate = f"{errors / calls * 100:.0f}%" if calls > 0 else "-"
                lines.append(f"| {tool} | {calls} | {errors} | {err_rate} |")
            lines.append("\n")

        # 证据
        if a["evidence"]:
            lines.append("<details>\n<summary>错误证据（日志片段）</summary>\n\n```\n")
            lines.append(a["evidence"] + "\n```\n\n</details>\n")

    # ── 7.3 全局工具调用统计 ──
    lines.append("\n### 7.3 全局工具调用统计\n")
    lines.append(
        "汇总所有失败 case 的工具调用情况，识别哪些工具使用最频繁、哪些出错率最高，"
        "以决定优化优先级。\n"
    )

    global_tool_stats = defaultdict(lambda: {"calls": 0, "errors": 0})
    for a in analyses:
        for tool, stats in a.get("tool_stats", {}).items():
            global_tool_stats[tool]["calls"] += stats["calls"]
            global_tool_stats[tool]["errors"] += stats["errors"]

    # 也统计通过 case 的工具调用（对比）
    passed_cases = [c for c in cases if is_pass(c["reward"])]
    pass_tool_stats = defaultdict(lambda: {"calls": 0, "errors": 0})
    for c in passed_cases:
        result_dir = Path(c["result_dir"])
        session_text = read_text(result_dir / "session.log", "")
        for m in TOOL_CALL_RE.finditer(session_text):
            pass_tool_stats[m.group(1)]["calls"] += 1
        for m in tool_error_re_find(session_text):
            pass_tool_stats[m]["errors"] += 1

    all_tools = sorted(set(list(global_tool_stats.keys()) + list(pass_tool_stats.keys())))
    lines.append("| 工具 | 失败 case 调用 | 失败 case 出错 | 通过 case 调用 | 通过 case 出错 |")
    lines.append("| ---- | -------------:| -------------:| -------------:| -------------:|")
    for tool in all_tools:
        f_calls = global_tool_stats[tool]["calls"]
        f_errors = global_tool_stats[tool]["errors"]
        p_calls = pass_tool_stats[tool]["calls"]
        p_errors = pass_tool_stats[tool]["errors"]
        lines.append(f"| {tool} | {f_calls} | {f_errors} | {p_calls} | {p_errors} |")
    lines.append("\n")

    # ── 7.4 Skills/Tools 优化建议 ──
    lines.append("\n### 7.4 Skills/Tools 优化建议\n")
    lines.append(
        "基于错误分类和工具统计，提出以下具体优化方向，按预期收益排序：\n"
    )

    suggestions = []

    # 轮次耗尽
    if cat_counter.get("turn_limit", 0) > 0:
        cases_str = ", ".join(
            f"{a['case']['version']}/{a['case']['name']}" for a in analyses
            if a["category"] == "turn_limit"
        )
        suggestions.append(
            f"1. **减少无效轮次消耗** — 轮次耗尽的 case: {cases_str}。"
            "建议：增强 Agent 的全局规划能力，减少重复试探（如反复编译-报错-修补循环）；"
            "引入阶段性自检（每 20 轮检查进度），避免在死胡同中浪费轮次。"
        )

    # 编译错误
    if cat_counter.get("compilation_error", 0) > 0:
        cases_str = ", ".join(
            f"{a['case']['version']}/{a['case']['name']}" for a in analyses
            if a["category"] == "compilation_error"
        )
        suggestions.append(
            f"2. **提升代码编译可靠性** — 编译失败的 case: {cases_str}。"
            "建议：在 Agent 写入代码后自动触发语法检查（如 `gcc -fsyntax-only`、"
            "`python -m py_compile`），在提交前捕获语法错误；"
            "增强对 include/依赖管理的推理能力。"
        )

    # 运行时错误
    if cat_counter.get("runtime_error", 0) > 0:
        exc_types = Counter()
        for a in analyses:
            if a["category"] == "runtime_error":
                exc_types.update(PYTHON_EXCEPTION_RE.findall(
                    read_text(Path(a["case"]["result_dir"]) / "session.log", "")
                ))
        top_excs = ", ".join(f"{e}({c})" for e, c in exc_types.most_common(5))
        suggestions.append(
            f"3. **增强运行时调试能力** — 运行时错误 case: {top_excs}。"
            "建议：Agent 在运行代码后应主动检查退出码和 stderr，"
            "遇到异常时先定位根因（读 traceback）再修复，而非盲目重试。"
            "增加对常见 Python 异常模式（KeyError→检查字典键, ValueError→检查参数范围）"
            "的自动诊断 skill。"
        )

    # Verifier 拒绝 / 逻辑错误
    if cat_counter.get("verifier_rejection", 0) > 0 or cat_counter.get("logic_error", 0) > 0:
        vr_cases = [a for a in analyses if a["category"] == "verifier_rejection"]
        le_cases = [a for a in analyses if a["category"] == "logic_error"]
        desc_parts = []
        if vr_cases:
            desc_parts.append(
                "verifier 拒绝: " + ", ".join(f"{a['case']['name']}" for a in vr_cases)
            )
        if le_cases:
            desc_parts.append(
                "逻辑错误: " + ", ".join(f"{a['case']['name']}" for a in le_cases)
            )
        suggestions.append(
            f"4. **强化任务输出自验证** — {'; '.join(desc_parts)}。"
            "建议：Agent 在提交最终输出前，应自行运行 verifier 脚本（如果可获取）"
            "或按任务描述中的验收标准逐项检查输出格式、精度、完整性。"
            "增加一个通用的 'output_validation' skill，在 agent 认为完成后"
            "自动对照 task.toml 中的验证规则做预检。"
        )

    # 环境错误
    if cat_counter.get("environment_error", 0) > 0:
        suggestions.append(
            "5. **改善基础设施可靠性** — 容器启动/镜像/下载失败。"
            "建议：在 run_all_cases.py 中增加重试逻辑（最多 3 次），"
            "对 Docker pull / build 失败自动重试；"
            "预热常用镜像（提前 pull），减少运行时失败率。"
        )

    # Agent 崩溃
    if cat_counter.get("agent_crash", 0) > 0:
        suggestions.append(
            "6. **增强 Agent 健壮性** — Agent 进程异常退出。"
            "建议：在 psi-agent 的 session 循环中增加顶层 try/except，"
            "捕获未预期异常并尝试恢复（如重新初始化上下文）；"
            "记录崩溃时的完整 traceback 便于定位。"
        )

    # 工具使用效率
    bash_errors = global_tool_stats.get("bash", {}).get("errors", 0)
    bash_calls = global_tool_stats.get("bash", {}).get("calls", 0)
    if bash_calls > 0 and bash_errors / bash_calls > 0.3:
        suggestions.append(
            f"7. **降低 bash 工具错误率** — 当前失败 case 中 bash 错误率 "
            f"{bash_errors}/{bash_calls} ({bash_errors / bash_calls * 100:.0f}%)。"
            "建议：Agent 在执行 bash 命令前应预判可能的失败模式"
            "（路径不存在、权限不足、依赖缺失），先做快速检查再执行重操作。"
            "对长命令拆分为独立步骤，每步验证后再继续。"
        )

    if not suggestions:
        suggestions.append(
            "暂无具体优化建议（错误分类已覆盖但均为低频小问题）。"
        )

    for s in suggestions:
        lines.append(f"- {s}\n")

    lines.append("\n")
    return lines


def tool_error_re_find(session_text):
    """辅助函数：找出 session_text 中所有工具执行出错的工具名。"""
    results = []
    for m in re.finditer(
        r"Tool result \('(\w+)'\):.*?\[Exit code: [^0]\]",
        session_text, re.DOTALL
    ):
        results.append(m.group(1))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 报告主体
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_md(stats, meta):
    total = stats["total_cases"]
    pass_rate = stats["pass_cases"] / total * 100 if total > 0 else 0.0
    api_req = stats["api_req"]

    table_data = [
        ("完成 case 数 / 总数", f'{stats["completed_cases"]} / {total}'),
        ("通过 case 数 (reward=1)", str(stats["pass_cases"])),
        ("失败 case 数", str(stats["fail_cases"])),
        ("unknown case 数", str(stats["unknown_cases"])),
        ("总 reward", f'{stats["reward_sum"]:.2f} / {total}'),
        ("通过率", f"{pass_rate:.1f}%"),
        ("输入 token", f'{stats["total_input_tokens"]:,}'),
        ("输出 token", f'{stats["total_output_tokens"]:,}'),
        ("总 token", f'{stats["total_tokens"]:,}'),
        ("token 来源", stats["token_source_label"]),
        ("API 请求数", f"{api_req:,}"),
        ("底层模型", f'`{meta.get("model", "unknown")}`'),
        ("Agent 版本", f'`{meta.get("agent_version", "unknown")}`'),
    ]
    lines = ["\n## 一、综合打分\n", "| 指标 | 数值 |", "| ---- | ---- |"]
    for title, value in table_data:
        lines.append(f"| {title} | {value} |")
    lines.append("\n")
    return lines


def render_group_table(cases, group_key, group_values, label):
    lines = []
    lines.append(f"| {label} | 总数 | 通过 | 失败 | unknown | reward 和 | 总 token |")
    lines.append("| ---- | ----:| ----:| ----:| -------:| ---------:| -------------:|")
    for g_val in group_values:
        subset = [c for c in cases if c.get(group_key) == g_val]
        if not subset:
            continue
        total = len(subset)
        pass_cnt = sum(is_pass(c["reward"]) for c in subset)
        fail_cnt = sum(is_fail(c["reward"]) for c in subset)
        unk_cnt = sum(is_unknown(c["reward"]) for c in subset)
        reward_sum = sum(reward_value(c["reward"]) or 0.0 for c in subset)
        token_sum = sum(c.get("total_tokens", 0) for c in subset)
        lines.append(
            f"| {g_val} | {total} | {pass_cnt} | {fail_cnt} | {unk_cnt} | {reward_sum:.2f} | {token_sum:,} |"
        )
    lines.append("\n")
    return lines


def generate(output_path=None, manifest_path=None):
    manifest_src = Path(manifest_path) if manifest_path else MANIFEST_JSON
    manifest = json.loads(read_text(manifest_src, "{}"))
    metadata = json.loads(read_text(METADATA_JSON, "{}"))
    meta = manifest.pop("_meta", {}) if "_meta" in manifest else {}

    cases = []
    for key, item in manifest.items():
        name = item.get("name", key.split("/")[-1])
        version = item.get("version", "")
        md = metadata.get(key, {})
        domain = md.get("domain", item.get("domain", ""))
        difficulty = md.get("difficulty", item.get("difficulty", ""))
        tokens = estimate_tokens(name)
        cases.append({
            "order": item.get("order", 0),
            "key": key,
            "version": version,
            "name": name,
            "domain": domain,
            "difficulty": difficulty,
            "agent_status": item.get("agent_status", ""),
            "reward": item.get("reward", ""),
            "elapsed_sec": item.get("elapsed_sec", 0) or 0,
            "requests": tokens["requests"],
            "session_chars": tokens["session_chars"],
            "output_chars": tokens["output_chars"],
            "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
            "total_tokens": tokens["total_tokens"],
            "token_source": tokens["token_source"],
            "result_dir": item.get("result_dir", str(RESULTS_DIR / name)),
        })

    cases.sort(key=lambda x: x["order"])
    total_cases = len(cases)
    completed_cases = sum(1 for c in cases if not is_unknown(c["reward"]))
    pass_cases = sum(1 for c in cases if is_pass(c["reward"]))
    fail_cases = sum(1 for c in cases if is_fail(c["reward"]))
    unknown_cases = sum(1 for c in cases if is_unknown(c["reward"]))
    reward_sum = sum(reward_value(c["reward"]) or 0.0 for c in cases)
    total_tokens = sum(c["total_tokens"] for c in cases)
    total_input_tokens = sum(c["input_tokens"] for c in cases)
    total_output_tokens = sum(c["output_tokens"] for c in cases)
    api_req = sum(c["requests"] for c in cases)
    elapsed = meta.get("elapsed_sec", sum(c["elapsed_sec"] for c in cases))

    token_sources = set(c["token_source"] for c in cases)
    if token_sources == {"实测"}:
        token_source_label = "实测"
    elif token_sources == {"估算"}:
        token_source_label = "估算"
    else:
        token_source_label = "混合(实测+估算)"

    stats = {
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "pass_cases": pass_cases,
        "fail_cases": fail_cases,
        "unknown_cases": unknown_cases,
        "reward_sum": reward_sum,
        "total_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "api_req": api_req,
        "token_source_label": token_source_label,
    }

    lines = []
    lines.append("# TB 2.1/3.0 Benchmark 数据报告\n")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> Agent 版本：`{meta.get('agent_version', 'unknown')}`\n")
    lines.append(f"> 模型：`{meta.get('model', 'unknown')}`\n")
    lines.append(f"> 运行起止：{meta.get('start_time', '-')} ~ {meta.get('end_time', '-')}\n")
    lines.append(f"> 总耗时：{format_elapsed(elapsed)}\n")
    lines.append("\n---\n")

    # 一、综合打分
    lines += build_summary_md(stats, meta)

    # 二、按版本统计
    lines.append("\n## 二、按版本统计\n")
    lines += render_group_table(cases, "version", ["2.1", "3.0"], "版本")

    # 三、按难度统计
    lines.append("\n## 三、按难度统计\n")
    lines += render_group_table(cases, "difficulty", ["易", "中", "难"], "难度")

    # 四、按领域统计
    lines.append("\n## 四、按领域统计\n")
    domains = []
    for c in cases:
        if c["domain"] and c["domain"] not in domains:
            domains.append(c["domain"])
    lines += render_group_table(cases, "domain", domains, "领域")

    # 五、详细结果
    lines.append("\n## 五、详细结果\n")
    lines.append("| # | 版本 | 任务名 | 领域 | 难度 | 状态 | reward | 耗时 | 请求数 | 输入 token | 输出 token | 总 token | 日志 |")
    lines.append("|---|------|--------|------|------|------|--------|------|--------|------------|------------|----------|------|")
    for c in cases:
        result_dir = Path(c["result_dir"])
        log_links = " ".join([
            f"[session]({result_dir / 'session.log'})",
            f"[agent]({result_dir / 'agent_output.log'})",
            f"[verifier]({result_dir / 'verifier.log'})",
        ])
        reward_str = str(c["reward"]) if c["reward"] not in ("", None) else "unknown"
        lines.append(
            f"| {c['order']:02d} | {c['version']} | {c['name']} | {c['domain']} | {c['difficulty']} | "
            f"{c['agent_status']} | {reward_str} | {format_elapsed(c['elapsed_sec'])} | "
            f"{c['requests']} | {c['input_tokens']:,} | {c['output_tokens']:,} | {c['total_tokens']:,} | {log_links} |"
        )
    lines.append("\n")

    # 六、每个 case 的运行中间结果（折叠）
    lines.append("\n## 六、Case 运行中间结果\n")
    lines.append(
        "以下按 case 展示最近日志片段，便于后续调优。token 优先采用各 case `ai.log` 中 psi-agent 记录的"
        "**真实 usage**（每次 API 请求的 prompt/completion/total_tokens）；旧日志未记录 usage 时回退为估算"
        "（假设未开 compaction、每轮 prompt 含完整历史，累计输入 ≈ session_chars × requests / 2 / 4）。\n"
    )
    for c in cases:
        result_dir = Path(c["result_dir"])
        session_text = read_text(result_dir / "session.log")
        agent_text = read_text(result_dir / "agent_output.log")
        verifier_text = read_text(result_dir / "verifier.log")
        session_tail = tail_lines(session_text, LOG_TAIL_LINES)
        agent_tail = tail_lines(agent_text, LOG_TAIL_LINES)
        verifier_tail = tail_lines(verifier_text, LOG_TAIL_LINES)

        lines.append(f"\n### {c['order']:02d}. {c['version']}/{c['name']}\n")
        lines.append(f"- 状态：{c['agent_status']} | reward：{c['reward']} | 耗时：{format_elapsed(c['elapsed_sec'])}\n")
        lines.append(f"- 请求数：{c['requests']} | 输入 token：{c['input_tokens']:,} ({c['token_source']}) | 输出 token：{c['output_tokens']:,} | 总 token：{c['total_tokens']:,}\n")
        lines.append("\n<details>\n<summary>session.log 最近 30 行</summary>\n\n```\n")
        lines.append(session_tail + "\n```\n\n</details>\n")
        lines.append("\n<details>\n<summary>agent_output.log 最近 30 行</summary>\n\n```\n")
        lines.append(agent_tail + "\n```\n\n</details>\n")
        lines.append("\n<details>\n<summary>verifier.log 最近 30 行</summary>\n\n```\n")
        lines.append(verifier_tail + "\n```\n\n</details>\n")

    # 七、错误分析（新增）
    lines += build_error_analysis_md(cases)

    report = "\n".join(lines)

    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_report_{ts}.md"
    else:
        output_path = Path(output_path)
    output_path.write_text(report, encoding="utf-8")
    # 仅打印裸路径，供 run_benchmark.sh / trigger_benchmark.py 解析下载
    print(str(output_path))
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 manifest 生成评测报告（含错误分析）")
    parser.add_argument("--out", default=None,
                        help="输出 markdown 路径（默认：results 下时间戳命名）")
    parser.add_argument("--manifest", default=None,
                        help="manifest JSON 路径（默认：manifests/benchmark_manifest.json）")
    args = parser.parse_args()
    generate(args.out, manifest_path=args.manifest)
