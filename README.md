# psi-agent-benchmark

HaiTun（psi-agent）统一评测工具。TB 通过 Harbor 官方流程评测，直接产出官方 `result.json`。

| Benchmark | 测什么 | 执行方式 | 打分 |
|---|---|---|---|
| **TB 2.1** | 终端任务（容器内） | `harbor run` + psi-agent adapter | 官方 result.json（success, failure_tags, parser_results） |
| **TB 3.0** | 终端任务（容器隔离） | `harbor run` + psi-agent adapter | 官方 result.json（+ 多容器隔离审计） |
| **tau2** | 多轮对话任务 | 本地 adapter → tau2-bench | 多维（reward + db_check + assertion） |
| **GAIA** | 通用研究能力 | 本地 workspace | 连续 0.0-1.0（exact match） |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/hys070414/psi-agent-benchmark-merged.git
cd psi-agent-benchmark-merged
```

### 2. 安装 Harbor

```bash
uv tool install harbor
# 或
pip install harbor-ai
```

### 3. 配置环境

```bash
cp .env.example .env
# 编辑 .env，填入：
#   PSI_AI_API_KEY=你的智谱 API Key
#   PSI_AI_MODEL=glm-5.3-max
#   PSI_AI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
#   PSI_AGENT_REPO=https://github.com/genuineknowledge/psi-agent.git
#   PSI_AGENT_WORKSPACE=workspaces/terminal_bench
```

### 4. 配置本地环境（tau2/GAIA 需要）

```powershell
.\setup_local.ps1
# 自动安装 uv + Python 3.14 + psi-agent + tau2-bench + GAIA 数据
```

### 4. 运行评测

```bash
# 单个 benchmark
python benchmark.py run -b tb-2.1
python benchmark.py run -b tb-3.0
python benchmark.py run -b tau2 --subset balanced_50
python benchmark.py run -b gaia --subset level1_smoke

# 全部四条线
python benchmark.py run -b all --run-id unified-001

# 预览不执行
python benchmark.py run -b all --dry-run
```

### 5. 生成报告（每个 benchmark 独立）

```bash
python benchmark.py report -b tb-2.1 --run-id tb21-001
python benchmark.py report -b tb-3.0 --run-id tb30-001
python benchmark.py report -b tau2 --run-id tau2-balanced50
python benchmark.py report -b gaia --run-id gaia-level1
python benchmark.py report -b all --run-id unified-001
```

## CLI 速查

```bash
python benchmark.py run     -b <tb-2.1|tb-3.0|tau2|gaia|all>  [选项]
python benchmark.py report  -b <同上>                          --run-id <ID>
python benchmark.py list     -b <同上>
python benchmark.py schema   -b <同上>                          # 查看数据格式和字段
```

### 常用选项

| 选项 | 适用 | 说明 |
|---|---|---|
| `--cases, -c` | TB | 精确指定 case 名称（逗号分隔） |
| `--difficulties, -d` | TB | 按难度筛（易,中,难） |
| `--exclude` | TB | 排除某个 case |
| `--limit, -n` | TB/GAIA | 最多跑 N 个 |
| `--subset, -s` | tau2/GAIA | 子集名（如 quick_3, level1_smoke） |
| `--run-id` | 全部 | 运行 ID（`-b all` 时自动加后缀） |
| `--dry-run` | 全部 | 预览不执行 |
| `--no-report` | 全部 | 跳过报告生成 |

## 各 Benchmark 的数据格式

用 `python benchmark.py schema -b all` 查看完整 schema。

| Benchmark | 数据来源 | 独有字段 | 报告章节 |
|---|---|---|---|
| tb-2.1 | Harbor 官方 result.json | failure_tags, parser_results | 7 章（+错误分析） |
| tb-3.0 | Harbor 官方 result.json | + 容器隔离审计字段 | 8 章（+多容器隔离审计） |
| tau2 | tau2-bench | db_check_pass, assertion_pass, user_messages | 5 章（含失败快照） |
| gaia | GAIA scorer | score(0-1), is_correct, file_count, search_count | 5 章（按 level 分组） |

## 报告目录

```
reports/
├── tb-2.1/<run-id>/report.md       # TB 2.1 报告
├── tb-3.0/<run-id>/report.md       # TB 3.0 报告
├── tau2/<run-id>/report.md         # tau2 报告
└── gaia/<run-id>/report.md         # GAIA 报告
```

## 目录结构

```
├── benchmark.py               # 统一 CLI 入口
├── src/orchestrator.py         # 四线调度器
├── adapters/
│   ├── terminal_bench/
│   │   └── harbor_agent.py     # psi-agent → Harbor 适配器（TB 核心）
│   └── tau2/
│       └── psi_agent_adapter.py # psi-agent → tau2 桥接
├── setup_local.ps1             # 本地环境配置（tau2/GAIA）
├── setup.sh                    # 服务器初始化（harbor + Docker）
├── requirements.txt            # 合并依赖
│
├── bin/                        # 所有脚本（TB + tau2 + GAIA）
│   ├── run_all_cases.py        # TB 批量调用 harbor run
│   ├── generate_report.py      # TB 报告生成（读 Harbor result.json）
│   ├── fetch_cases.py          # TB case 拉取
│   ├── trigger_benchmark.py    # TB 远程触发
│   ├── fetch_report.py         # TB 报告拉取
│   ├── psi_agent_benchmark.py  # tau2/GAIA runner
│   ├── gaia_benchmark.py       # GAIA runner
│   └── generate_tau2_report.py # tau2 报告
├── config/                     # 所有配置（TB + tau2 + GAIA）
│   ├── benchmark.yaml
│   ├── case_metadata.json
│   ├── tau2_subsets.json
│   └── gaia_subsets.json
├── adapters/                   # tau2 适配器
│   └── tau2/psi_agent_adapter.py
│
├── workspaces/
│   ├── terminal_bench/         # TB workspace
│   └── gaia/                   # GAIA workspace
│
└── build_images.sh             # Docker 镜像构建
```

## 环境变量

### `.env`（远程 TB）

```env
TB_BENCH_HOST=your.server.ip
TB_BENCH_USER=root
TB_BENCH_KEY=~/.ssh/id_ed25519
TB_BENCH_WORKDIR=/root/psi-agent-benchmark
```

### `.env.local`（本地 tau2/GAIA，由 setup_local.ps1 生成）

```env
PSI_ROOT=C:\Users\...\psi-agent-main
TAU2_ROOT=external/tau2-bench
GAIA_DATA_ROOT=external/gaia-data
PSI_AI_PROVIDER=openai
PSI_AI_MODEL=glm-5.3-max
PSI_AI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

## 定时自动跑（可选）

在服务器上加 cron：

```cron
0 6 * * * cd /root/psi-agent-benchmark && bash bin/run_benchmark.sh >> pilot_results/cron.log 2>&1
```

发到 GitHub reports 分支：设 `GITHUB_TOKEN` 后 `bash bin/auto_bench.sh`。
