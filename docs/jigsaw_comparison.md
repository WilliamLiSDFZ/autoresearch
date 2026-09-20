# Jigsaw：baseline / analogy 配对实验

两个 Job 各运行最多 **6 小时**，自动安装基础工具、执行 `uv sync`，然后等待你进入 Pod 手动运行 `claude`。baseline 执行 `program.md`；analogy 执行 `program-analogy.md`，在 draft 和每次 improve 前检索，且禁止修改 analogy 代码。

## 隔离方式与时限

**先创建并 checkout 不同 branch，也不能隔离同一个工作目录。** 同一目录只有一份当前 HEAD、工作文件和 Git index；B checkout 后，A 后续操作看到的也是切换后的目录，未提交文件也不会因分支名不同而隔离。因此两组各有一份独立 clone，agent 在自己的目录里首先创建并切换新分支，再修改代码。此处不用需要额外挂载主仓库的 linked worktree。

两组隔离 repo、`.venv`、结果与 `/root` 下的 Claude 会话；共享只读 prepared 数据和 Claude Code 安装文件。仅 analogy 挂载知识库。不要把整个 PVC 根目录再挂到这两个 Pod。

两份 YAML 均设置：

```yaml
activeDeadlineSeconds: 21600
backoffLimit: 0
```

6 小时从 Job 的 `.status.startTime` 开始，**包括调度、安装依赖和等待人工进入的时间**，并非 Claude 启动后还有完整 6 小时。Kubernetes 到期终止 Pod，Job 通常显示 `Failed / DeadlineExceeded`，不会自动重跑。[Kubernetes Job 时限说明](https://kubernetes.io/docs/concepts/workloads/controllers/job/#job-termination-and-cleanup)

这一轮比较相同 Job 总时限下的最佳已完成验证分数；安装或人工等待不同，会使两组实际研究时间不同。尽量同时准备、及时启动并记录时间。analogy 的额外模型用量另行记录，这不是相同 token 或 API 成本的对照。

## 1. 准备目录与配置

默认 namespace `ecepxie`、PVC `yuze-li-vol`、单张 `NVIDIA-A40`、8 CPU、48Gi 内存。修改硬件时，两份 YAML 的资源、`nodeSelector` 和 `AUTORESEARCH_RESOURCES` 一起改。

先由你把当前代码和文档提交到 Jigsaw 任务分支，并同步到 PVC 的 `/workspace/autoresearch`。下面 clone 只包含已提交内容。两组从同一任务 commit 开始，不复制已跑好的 baseline 代码、结果或 `.venv`。

在原来挂载整个 PVC 的 dev Pod 中执行：

```bash
set -e
PAIR_ROOT=/workspace/autoresearch-pairs/jubias-pair-001
TASK_COMMIT=$(git -C /workspace/autoresearch rev-parse codex/jigsaw-unintended-bias)
for ARM in baseline analogy; do
  mkdir -p "$PAIR_ROOT/$ARM/home"
  git clone --no-local --single-branch --branch codex/jigsaw-unintended-bias \
    /workspace/autoresearch "$PAIR_ROOT/$ARM/repo"
  git -C "$PAIR_ROOT/$ARM/repo" checkout --detach "$TASK_COMMIT"
done
```

这里仅固定两个 clone 的起点；**run 分支由之后启动的 agent 创建**，格式为 `run/YYYYMMDD_HHMMSS-jigsaw-unintended-bias-in-toxicity-classification-baseline` 或 `...-analogy`。除首次创建并切换分支外，program 仍不允许 agent commit、push 或 reset。

核对 YAML 中的 PVC 路径。`subPath` 相对 PVC 根目录，不带 `/workspace/`：

| 内容 | 默认 PVC subPath | Pod 内路径 |
| --- | --- | --- |
| 固定数据 | `autoresearch/results/jigsaw-data` | `/data/jigsaw-prepared` |
| 任务说明 | `data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md` | `/data/task/description.md` |
| KB，仅 analogy | `Agentic_Knowledge_Base/output/paper_corpus` | `/data/paper-corpus` |
| 已安装的 Claude Code | `home/.local` | `/workspace/home/.local` |

Claude 路径按原 dev Pod 的 `HOME=/workspace/home` 和原生安装布局配置：`CLAUDE_BIN=/workspace/home/.local/bin/claude`。Job 保留原路径挂载，避免已有绝对符号链接失效。若 `command -v claude` 或 `readlink -f` 显示其他安装位置，请对应修改两份 Job 的 `CLAUDE_BIN` 与只读安装目录挂载；**不会重新下载 Claude Code**。安装文件共享，会话和配置仍在各自 `/root`，可能需要分别登录一次。

数据提前准备一次，实验时只读；KB 目录需有 `records.jsonl` 和 `manifest.json`。实际路径不同就改 YAML。analogy Job 中的 `ANALOGY_CORPUS_DIR` 配置 KB 挂载位置；`ANALOGY_MODEL` / `ANALOGY_BASE_URL` 暂沿用 MLEvolve 的 `gpt-5.6-sol` / `http://cliproxy:8317/v1`，按实际接口修改。

下一对实验请同时换 Job 名、pair label、`RUN_TAG` 及 repo/home 的 `subPath`，使用新目录。实验期间不要从 dev Pod 修改共享固定数据、语料或 Claude 安装。

## 2. 创建 Job，等待安装完成

在本项目目录执行：

```bash
kubectl --context nautilus apply -f k8s/job-jigsaw-baseline.yaml
kubectl --context nautilus apply -f k8s/job-jigsaw-analogy.yaml
kubectl --context nautilus -n ecepxie get pods -l experiment-pair=jubias-pair-001
kubectl --context nautilus -n ecepxie logs -f job/autoresearch-jubias-pair-001-baseline
```

Job 内已依次执行 `apt-get update`、安装 `git curl ca-certificates gcc`、安装固定版本 uv、`uv sync --locked --python 3.10`、检查已有 Claude 可执行文件。看到 `Setup complete` 或 Pod `Ready` 后进入；安装失败会保留错误日志并退出。

```bash
kubectl --context nautilus -n ecepxie exec -it job/autoresearch-jubias-pair-001-baseline -- bash
# 另一个终端进入 analogy：
kubectl --context nautilus -n ecepxie exec -it job/autoresearch-jubias-pair-001-analogy -- bash
```

## 3. 直接启动 Claude，提供 prompt

两个 Pod 都在 `/workspace/autoresearch`。可以先检查 GPU：

```bash
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

只有 analogy Pod 需要额外设置检索 API key；它与 Claude 登录不同，不写入 YAML、prompt 或 Git：

```bash
read -r -s -p 'Analogy API key: ' ANALOGY_API_KEY
export ANALOGY_API_KEY
```

两组使用相同主模型、权限和思考设置，启动全新会话，不恢复旧实验：

```bash
claude
```

baseline prompt：

```text
阅读并执行 program.md，开始 Jigsaw baseline 实验。
这是本组独立的工作目录；任何修改之前先创建并 checkout 新的 run 分支。
Job 自启动起限时6小时，不要从现在重新计算6小时。
完全不使用 analogy agent、知识库、类比缓存或报告；不 commit 或 push。
```

analogy prompt：

```text
阅读并执行 program-analogy.md，开始 Jigsaw analogy 实验。
这是本组独立的工作目录；任何修改之前先创建并 checkout 新的 run 分支。
Job 自启动起限时6小时，不要从现在重新计算6小时。
必须在 draft 和每次 improve 前使用 analogy agent，不修改其代码；不 commit 或 push。
```

如需让 agent 按剩余时间规划，可查看各 Job 的实际开始时间，把“该开始时间 + 6 小时”的 UTC 截止时间补进 prompt；无需计时脚本：

```bash
kubectl --context nautilus -n ecepxie get jobs -l experiment-pair=jubias-pair-001 \
  -o custom-columns=NAME:.metadata.name,START:.status.startTime,LIMIT_SECONDS:.spec.activeDeadlineSeconds
```

## 4. 查看结果

Pod 到期后回到 dev Pod，从以下 PVC 目录读取结果：

```text
/workspace/autoresearch-pairs/jubias-pair-001/baseline/repo/results/jubias-pair-001-baseline/
/workspace/autoresearch-pairs/jubias-pair-001/analogy/repo/results/jubias-pair-001-analogy/
```

两组共用中性 `experiment_artifacts.py` 保存训练前源码并绑定完成后的指标、配置、日志；baseline 不调用 analogy 的任何命令。helper 只负责结果绑定，不负责限时。比较 `source.json.execution_status == "completed"` 且 `completed_at_utc` 早于实际 Job 截止时间的候选；未完成或被中断的候选不计分。没有完成候选就报告无有效结果。

保存 Job 的开始/结束状态、两组起始 Git SHA、prepared ID、GPU/环境/主模型配置、`results.tsv`、最佳候选以及 analogy 的检索记录。一次配对适合先确认流程，之后再用预先约定的种子做重复实验。

确认结果后可删除 Job，PVC 上的文件会保留：

```bash
kubectl --context nautilus -n ecepxie delete job autoresearch-jubias-pair-001-baseline autoresearch-jubias-pair-001-analogy
```
