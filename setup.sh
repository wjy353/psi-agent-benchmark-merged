#!/usr/bin/env bash
# ============================================================================
# setup.sh — 在服务器上初始化 psi-agent + Terminal-Bench 评测环境
#
# 用法:
#   bash setup.sh                          # 使用默认配置
#   TB_BENCH_WORKDIR=/opt/tb bash setup.sh # 自定义工作目录
#   PSI_AGENT_REF=feature-x bash setup.sh  # 测试指定分支
# ============================================================================
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────
WORKDIR="${TB_BENCH_WORKDIR:-$HOME/psi-agent-benchmark}"
PSI_DIR="$WORKDIR/psi-agent"
PSI_AGENT_REF="${PSI_AGENT_REF:-main}"
PSI_AGENT_REPO="${PSI_AGENT_REPO:-https://github.com/genuineknowledge/psi-agent.git}"

echo "[setup] workdir: $WORKDIR"
echo "[setup] psi-agent ref: $PSI_AGENT_REF"

mkdir -p "$WORKDIR"

# ── 1. 安装 Python 依赖 ──────────────────────────────────────────────────
echo "[setup] installing Python dependencies..."
pip3 install -r "$WORKDIR/requirements.txt" 2>/dev/null \
    || echo "[setup] pip3 not available, skipping Python dependencies"

# ── 2. 拉取 psi-agent 并切换到指定版本 ───────────────────────────────────
if [ ! -d "$PSI_DIR/.git" ]; then
    echo "[setup] cloning psi-agent from $PSI_AGENT_REPO..."
    git clone "$PSI_AGENT_REPO" "$PSI_DIR"
fi

cd "$PSI_DIR"
git fetch origin
git checkout "$PSI_AGENT_REF"
git pull origin "$PSI_AGENT_REF" || true
cd "$WORKDIR"

echo "[setup] psi-agent commit: $(cd "$PSI_DIR" && git rev-parse --short HEAD)"

# ── 3. 部署脚本和配置到 psi-agent 目录 ──────────────────────────────────
echo "[setup] deploying scripts to psi-agent..."
# 安装 psi-agent 包（editable 模式，确保 import 可用）
cd "$PSI_DIR"
uv pip install -e . 2>/dev/null || pip install -e . 2>/dev/null || echo "[setup] WARNING: could not install psi-agent package"
cd "$WORKDIR"

cp "$WORKDIR/run_all_cases.py"   "$PSI_DIR/run_all_cases.py"
cp "$WORKDIR/generate_report.py" "$PSI_DIR/generate_report.py"
cp "$WORKDIR/fetch_cases.py"     "$PSI_DIR/fetch_cases.py"
cp "$WORKDIR/build_images.sh"    "$PSI_DIR/build_images.sh"
chmod +x "$PSI_DIR/build_images.sh"

# 部署容器管理模块（复制到 psi-agent 的 src/，与 psi_agent/ 包共存）
cp "$WORKDIR/src/container.py" "$PSI_DIR/src/container.py"
echo "[setup] deployed container.py to $PSI_DIR/src/container.py"
cp "$WORKDIR/src/case_source.py" "$PSI_DIR/src/case_source.py"
echo "[setup] deployed case_source.py to $PSI_DIR/src/case_source.py"

mkdir -p "$WORKDIR/manifests"
cp "$WORKDIR/config/case_metadata.json" "$WORKDIR/manifests/case_metadata.json"

cp "$WORKDIR/bin/run_benchmark.sh" "$PSI_DIR/run_benchmark.sh"
chmod +x "$PSI_DIR/run_benchmark.sh"

# ── 3. 部署容器版 workspace — 工具通过 PSI_PILOT_CONTAINER env 操作 Docker 容器
rm -rf "$PSI_DIR/examples/terminal_bench"
cp -r "$WORKDIR/workspaces/terminal_bench" "$PSI_DIR/examples/terminal_bench"
cp "$WORKDIR/src/container.py" "$PSI_DIR/src/container.py"
echo "[setup] deployed container workspace to $PSI_DIR/examples/terminal_bench"

# ── 4. 环境变量配置 ──────────────────────────────────────────────────────
if [ ! -f "$WORKDIR/.env" ]; then
    cp "$WORKDIR/config/.env.example" "$WORKDIR/.env"
    echo "[setup] created $WORKDIR/.env — please edit it with real credentials"
fi
# 复制到 psi-agent 目录，确保 run_all_cases.py 能直接读取
cp "$WORKDIR/.env" "$PSI_DIR/.env"

# ── 5. 预检：harbor / docker ─────────────────────────────────────────────
HARBOR_BIN="${TB_HARBOR_BIN:-harbor}"
if command -v "$HARBOR_BIN" &>/dev/null; then
    echo "[setup] harbor: $($HARBOR_BIN --version 2>&1 || echo 'found')"
else
    echo "[setup] WARNING: harbor not found — install Terminal-Bench CLI to build images"
    echo "[setup]   pip install terminal-bench"
fi

if docker ps &>/dev/null; then
    echo "[setup] docker: OK"
else
    echo "[setup] WARNING: docker not accessible — ensure user is in docker group"
fi

echo ""
echo "[setup] done. Next steps:"
echo "  1. edit $WORKDIR/.env"
echo "  2. bash $WORKDIR/build_images.sh"
echo "  3. bash $PSI_DIR/run_benchmark.sh"
