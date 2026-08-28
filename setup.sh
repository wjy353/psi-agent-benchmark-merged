#!/usr/bin/env bash
# ============================================================================
# setup.sh — 在服务器上初始化 Harbor + Terminal-Bench 评测环境
#
# harbor run 管理容器生命周期、verifier 执行和 result.json 生成。
# 本脚本只负责：安装 harbor、拉取 case 元数据、配置 .env。
#
# 用法:
#   bash setup.sh                          # 使用默认配置
#   TB_BENCH_WORKDIR=/opt/tb bash setup.sh # 自定义工作目录
# ============================================================================
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────
WORKDIR="${TB_BENCH_WORKDIR:-$HOME/psi-agent-benchmark}"
PSI_AGENT_REF="${PSI_AGENT_REF:-main}"
PSI_AGENT_REPO="${PSI_AGENT_REPO:-https://github.com/genuineknowledge/psi-agent.git}"
MODEL="${PSI_AI_MODEL:-glm-5.3-max}"

echo "[setup] workdir: $WORKDIR"
echo "[setup] model: $MODEL"

mkdir -p "$WORKDIR"

# ── 1. 安装 Harbor ────────────────────────────────────────────────────────
echo "[setup] installing harbor..."
if ! command -v harbor &>/dev/null; then
    pip3 install harbor-ai 2>/dev/null \
        || pip install harbor-ai 2>/dev/null \
        || (curl -LsSf https://astral.sh/uv/install.sh | sh && uv tool install harbor)
fi
echo "[setup] harbor: $(harbor --version 2>&1 || echo 'NOT FOUND')"

# ── 2. 安装 Python 依赖 ──────────────────────────────────────────────────
echo "[setup] installing Python dependencies..."
pip3 install -r "$WORKDIR/requirements.txt" 2>/dev/null \
    || echo "[setup] WARNING: some Python dependencies may be missing"

# ── 3. 拉取 case 元数据 ──────────────────────────────────────────────────
echo "[setup] fetching case metadata..."
cd "$WORKDIR"
python3 bin/fetch_cases.py 2>/dev/null \
    || echo "[setup] WARNING: could not fetch case metadata (run bin/fetch_cases.py manually)"

# ── 4. 部署 workspace ─────────────────────────────────────────────────────
mkdir -p "$WORKDIR/workspaces/terminal_bench"
echo "[setup] workspace at $WORKDIR/workspaces/terminal_bench"

# ── 5. 环境变量配置 ──────────────────────────────────────────────────────
if [ ! -f "$WORKDIR/.env" ]; then
    cat > "$WORKDIR/.env" << ENV_EXAMPLE
PSI_AI_PROVIDER=openai
PSI_AI_MODEL=$MODEL
PSI_AI_API_KEY=your-api-key-here
PSI_AI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
PSI_AGENT_REPO=$PSI_AGENT_REPO
PSI_AGENT_REF=$PSI_AGENT_REF
PSI_AGENT_WORKSPACE=$WORKDIR/workspaces/terminal_bench
ENV_EXAMPLE
    echo "[setup] created $WORKDIR/.env — please edit it with real credentials"
fi

# ── 6. 预检：Docker ──────────────────────────────────────────────────────
if docker ps &>/dev/null; then
    echo "[setup] docker: $(docker --version)"
    echo "[setup] docker daemon: running"
else
    echo "[setup] WARNING: docker daemon not running — harbor needs Docker"
fi

echo ""
echo "[setup] done. Next steps:"
echo "  1. Edit .env with your API key"
echo "  2. python benchmark.py run -b tb-3.0 --limit 2 --run-id smoke"
echo "  3. python benchmark.py report -b tb-3.0 --run-id smoke"
