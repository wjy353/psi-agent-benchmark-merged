# psi-agent-benchmark

HaiTun（psi-agent）统一评测工具。支持四条独立评测线，每条线单独生成报告：

| Benchmark | 测什么 | 执行方式 | 打分 |
|---|---|---|---|
| **TB 2.1** | 终端任务（容器内） | 远程 SSH → Docker | 二值 0/1（verifier=same） |
| **TB 3.0** | 终端任务（容器隔离） | 远程 SSH → Docker | 二值 0/1（verifier=separate） |
| **tau2** | 多轮对话任务 | 本地 adapter → tau2-bench | 多维（reward + db_check + assertion） |
| **GAIA** | 通用研究能力 | 本地 workspace | 连续 0.0-1.0（exact match） |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/hys070414/psi-agent-benchmark-merged.git
cd psi-agent-benchmark-merged
```

### 2. 配置远程服务器（TB 2.1/3.0 需要）

```bash
cp .env.example .env
# 编辑 .env，填入服务器 IP 和 SSH 私钥路径
```

### 3. 配置本地环境（tau2/GAIA 需要）

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

| Benchmark | 独有字段 | 报告章节 |
|---|---|---|
| tb-2.1 | verifier_stdout, verifier_stderr | 6 章（总分/难度/领域/详情/中间结果/错误分析） |
| tb-3.0 | + docker_cp_path, overlay_image, bind_mount_path | 7 章（多容器隔离审计） |
| tau2 | db_check_pass, assertion_pass, user_messages, tool_calls | 5 章（含失败快照） |
| gaia | score(0-1), is_correct, file_count, search_count | 5 章（按 level 分组） |

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
├── setup_local.ps1             # 本地环境配置（tau2/GAIA）
├── setup.sh                    # 服务器初始化（TB）
├── requirements.txt            # 合并依赖
│
├── bin/                        # 远程触发脚本（TB）
├── config/                     # TB 配置（case_metadata.json, benchmark.yaml）
├── configs/                    # tau2/GAIA 子集配置
├── scripts/                    # tau2/GAIA 脚本
├── adapters/                   # tau2 适配器
│
├── workspaces/
│   ├── terminal_bench/         # TB workspace（bash/read/write/edit 工具）
│   └── gaia/                   # GAIA workspace
│
├── run_all_cases.py            # TB 评测主控
├── generate_report.py          # TB 报告生成
├── fetch_cases.py              # TB case 拉取
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
PSI_AI_MODEL=deepseek-chat
PSI_AI_BASE_URL=https://api.deepseek.com/v1
```

## 定时自动跑（可选）

在服务器上加 cron：

```cron
0 6 * * * cd /root/psi-agent-benchmark && bash bin/run_benchmark.sh >> pilot_results/cron.log 2>&1
```

发到 GitHub reports 分支：设 `GITHUB_TOKEN` 后 `bash bin/auto_bench.sh`。
