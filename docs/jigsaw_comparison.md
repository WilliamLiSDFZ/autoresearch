# Jigsaw：baseline / analogy 配对实验

两个 Job 各运行最多 **6 小时**，自动安装基础工具，激活并检查 dev Pod 已准备的 `/workspace/autoresearch/.venv`，然后等待你进入 Pod 手动运行 `claude`。Job 不再安装 uv、执行 `uv sync` 或下载 Python/PyTorch 依赖。baseline 执行 `program.md`；analogy 执行 `program-analogy.md`，在 draft 和每次 improve 前检索，且禁止修改 analogy 代码。

## 隔离方式与时限

两组 Job 挂载同一份主仓库 `/workspace/autoresearch`。agent 开始实验时用 `git worktree add -b` 创建本组的 run 分支和工作目录，之后只在自己的 worktree 修改、训练和保存结果。每个 worktree 有独立的 HEAD、index 和工作文件，因此不会因为另一组切换分支而改变本组文件；不再提前 clone，也不在主仓库执行 checkout。

两组各有独立的 `/root` 保存 Claude 会话和运行缓存，共享只读的 `/workspace/autoresearch/.venv`。`AUTORESEARCH_VENV` 和 `VIRTUAL_ENV` 均指向这份预装环境；worktree 中不创建新的 `.venv`。prepared 数据在 `/data/jigsaw-prepared` 和主仓库内的原路径均只读，Claude Code 安装文件也只读共享。仅 analogy 挂载知识库。主仓库及其 `.git` 必须在两个 Pod 中保留相同绝对路径，实验期间不要移动它们。

worktree 共享 Git 元数据，且两组都能访问主仓库挂载内的目录，这是工作目录隔离，并非文件访问权限隔离。program 明确禁止读取另一组或历史实验的代码、结果与报告，也禁止实验期间 commit、push、reset、切换分支或删除 worktree。

两份 YAML 均设置：

```yaml
activeDeadlineSeconds: 21600
backoffLimit: 0
```

6 小时从 Job 的 `.status.startTime` 开始，**包括调度、安装依赖和等待人工进入的时间**，并非 Claude 启动后还有完整 6 小时。Kubernetes 到期终止 Pod，Job 通常显示 `Failed / DeadlineExceeded`，不会自动重跑。[Kubernetes Job 时限说明](https://kubernetes.io/docs/concepts/workloads/controllers/job/#job-termination-and-cleanup)

这一轮比较相同 Job 总时限下的最佳已完成验证分数；安装或人工等待不同，会使两组实际研究时间不同。尽量同时准备、及时启动并记录时间。analogy 的额外模型用量另行记录，这不是相同 token 或 API 成本的对照。

**不需要守着收尾。** 六小时到期自动停止；如果整轮任务提前完成，program 要求 agent 先保存结果和 `summary.md`，最后执行 `touch /tmp/autoresearch-finished`。Job 的主进程检测到标记后以 0 退出，Job 成为 `Complete`，释放 GPU。无法继续的失败则写 `/tmp/autoresearch-failed`，以 1 退出并成为 `Failed`。普通训练失败可以继续调试，不应提前停整轮研究。

这两个标记只属于当前 Pod，不需要 Kubernetes API 权限。Claude 回复完一段话、关闭交互界面或结束单次训练，都不会自动等同于整轮研究完成；正常无提前终止条件时仍跑到六小时。Job/Pod 对象会保留供查看状态，结果保留在 PVC，停止运行不等于删除 Job 对象。

## 1. 准备目录与配置

默认 namespace `ecepxie`、PVC `yuze-li-vol`、单张 NVIDIA GPU、8 CPU、32Gi 内存。两份 Job 请求 `nvidia.com/gpu: 1`，使用 `nvidia.com/gpu.compute.major > 6` 保留计算能力 ≥7.0 的卡，不要求具体型号或最低显存。当前共享环境固定为 Torch 2.9.1+cu128，支持范围从 Volta 开始；1080 Ti、TITAN Xp 等 Pascal 卡不在范围内。[PyTorch 2.9.1 架构检查源码](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/cuda/__init__.py)

该规则可纳入 V100 16GB、T4、RTX 2080 Ti 及更新架构的卡。集群查询中还有一台 `NVIDIA-L40S` 缺少 compute 标签，因此增加了“compute 标签不存在且型号为 L40S”的后备规则；不会把未知型号或明确标为 6.x 的卡放进来。两条规则都保留故障节点排除。调度仍受资源、taint 和权限限制；部分特殊 GPU 使用独立资源名称，`nvidia.com/gpu: 1` 不能覆盖所有 GPU 资源池。[NRP GPU 资源说明](https://nrp.ai/documentation/userdocs/running/gpu-pods/)

两份 Job 暂时排除 `nautilus-ext-gpu01.fullerton.edu`，该节点在 2026-09-20 连续出现 GPU admission 和 CSI 驱动故障。确认节点修复后再由你同步调整两份配置。

两组可能分到不同 GPU，不能视为严格相同硬件的对照。开始前用 `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` 记录实际型号和显存，并按分到的卡调整模型、batch size 和精度；严格比较时应配对相同型号的运行。`AUTORESEARCH_RESOURCES` 描述的是调度范围，不是实际分配型号。架构标签只是初筛，实际设备、驱动和固定 PyTorch 的组合还必须通过启动时的 CUDA 运算检查；失败时退出，禁止改用 CPU 继续这组 GPU 实验。

先由你把当前代码和文档提交到 Jigsaw 任务分支，并同步到 PVC 的 `/workspace/autoresearch`，让主仓库停在本轮起始 commit。worktree 只包含该 commit 的已提交文件，不包含未提交修改或未跟踪文件；起始 `train.py` 应是共同的上游模板，不是已优化的 baseline。保留已有 prepared 数据，无需复制结果或 `.venv`。

在原来挂载整个 PVC 的 dev Pod 中执行：

```bash
set -e
test -d /workspace/autoresearch/.git
test -f /workspace/autoresearch/pyproject.toml
test -z "$(git -C /workspace/autoresearch status --porcelain)"
git -C /workspace/autoresearch rev-parse HEAD
PAIR_ROOT=/workspace/autoresearch-pairs/jubias-pair-001
for ARM in baseline analogy; do
  mkdir -p "$PAIR_ROOT/$ARM/home"
done
```

**run 分支和 worktree 都由之后启动的 agent 创建**，无需提前建空目录。分支格式为 `run/YYYYMMDD_HHMMSS-jigsaw-unintended-bias-in-toxicity-classification-baseline` 或 `...-analogy`，开始时间使用 UTC。Job 通过 Downward API 读取当前 Pod UID，生成 `RUN_TAG` 和 `AUTORESEARCH_REPO_DIR`；每次重建 Pod 都会自动换目录，即使沿用同一个 Job 名也不会与旧实验冲突：[Pod UID](https://kubernetes.io/docs/concepts/workloads/pods/downward-api/)、[环境变量展开顺序](https://kubernetes.io/docs/tasks/inject-data-application/define-interdependent-environment-variables/)。

```text
/workspace/autoresearch/worktrees/jubias-pair-001-baseline-<baseline-pod-uid>
/workspace/autoresearch/worktrees/jubias-pair-001-analogy-<analogy-pod-uid>
```

两份 Job 各自把主仓库 HEAD 记入 `/tmp/autoresearch-start-commit`，agent 据此创建 worktree，并检查主仓库仍干净且 HEAD 未变。两个 Job 启动前至本轮结束，保持主仓库不变；启动 Claude 前核对两个 Pod 记录的 SHA 相同。同一个 Pod 只启动一轮实验；重复执行初始化会报错停止，不复用或覆盖已有目录。创建 worktree 失败后不得进入旧目录继续操作，结果目录和 TSV 表头也只能创建一次。旧 worktree 和可能残留的空分支保留，由你之后处理。

核对 YAML 中的 PVC 路径。`subPath` 相对 PVC 根目录，不带 `/workspace/`：

| 内容 | 默认 PVC subPath | Pod 内路径 |
| --- | --- | --- |
| 主仓库，含两组 worktree | `autoresearch` | `/workspace/autoresearch` |
| 本组会话与运行缓存 | `autoresearch-pairs/jubias-pair-001/{baseline或analogy}/home` | `/root` |
| 已安装的 Python 环境，只读 | `autoresearch/.venv` | `/workspace/autoresearch/.venv` |
| dev Pod 的 uv-managed Python，只读 | `.local/share/uv/python` | `/workspace/.local/share/uv/python` |
| 固定数据 | `autoresearch/results/jigsaw-data` | `/data/jigsaw-prepared` |
| 固定数据的主仓库路径，同样只读 | `autoresearch/results/jigsaw-data` | `/workspace/autoresearch/results/jigsaw-data` |
| 任务说明 | `data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md` | `/data/task/description.md` |
| KB，仅 analogy | `Agentic_Knowledge_Base/output/paper_corpus` | `/data/paper-corpus` |
| 已安装的 Claude Code | `home/.local` | `/workspace/home/.local` |

Claude 路径按原 dev Pod 的 `HOME=/workspace/home` 和原生安装布局配置：`CLAUDE_BIN=/workspace/home/.local/bin/claude`。Job 保留原路径挂载，避免已有绝对符号链接失效。若 `command -v claude` 或 `readlink -f` 显示其他安装位置，请对应修改两份 Job 的 `CLAUDE_BIN` 与只读安装目录挂载；**不会重新下载 Claude Code**。安装文件共享，会话和配置仍在各自 `/root`，可能需要分别登录一次。

**复用已有 venv。** dev Pod 和 Job 使用相同镜像。dev Pod 的 `UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python` 已持久化；Job 在同一绝对路径只读挂载它，保留 `.venv/bin/python` 的符号链接目标。仅有 `.venv` 目录而缺少它引用的 Python 解释器，不能正常运行。

已在 dev Pod 执行过完整的 `uv sync` 且环境对应当前 `uv.lock`，就不用再安装。在 dev Pod 中可先检查：

```bash
cd /workspace/autoresearch
test -f .venv/bin/activate
readlink -f .venv/bin/python
.venv/bin/python -c 'import sys, numpy, pandas, torch, openai, rank_bm25, nltk, jsonschema; print(sys.prefix, sys.version, torch.__version__, openai.__version__)'
```

只有环境缺失或依赖已变化时，才在**没有实验使用共享环境期间**于 dev Pod 一次性执行以下命令。先同步当前项目和 `uv.lock`，再安装完整训练和 analogy 依赖；不能只装 `--only-group analogy`：

```bash
cd /workspace/autoresearch || exit 1
UV_PROJECT_ENVIRONMENT=/workspace/autoresearch/.venv \
UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python \
UV_LINK_MODE=copy \
uv sync --locked --python 3.10 --group analogy
```

`openai` 已在 analogy 依赖组中；出现 `ModuleNotFoundError` 通常表示现有环境缺少该组，需要上述完整同步，不是让实验 agent 临时安装单个包。若解释器引用临时 `/root` 路径，应先在 dev Pod 使用上述持久化 Python 目录重新准备环境。不要移动已创建的 venv；Job 不会自动修复或下载缺失依赖。两组运行期间，也不要从仍可写 PVC 的 dev Pod 更新共享环境。

数据提前准备一次，实验时只读。当前数据集位于 PVC 的 `autoresearch/results/jigsaw-data/seed-42`，因此 `AUTORESEARCH_JIGSAW_DIR=/data/jigsaw-prepared/seed-42`；挂载根目录仍为 `/data/jigsaw-prepared`。变量必须直接指向包含 `manifest.json`、三个 CSV 和 `split.npz` 的目录，不能指向它的父目录。在 dev Pod 可执行 `.venv/bin/python prepare.py verify --prepared-dir /workspace/autoresearch/results/jigsaw-data/seed-42` 验证已有数据，无须重做 prepare。KB 目录需有 `records.jsonl` 和 `manifest.json`。实际路径不同就改 YAML。analogy Job 中的 `ANALOGY_CORPUS_DIR` 配置 KB 挂载位置；`ANALOGY_MODEL` / `ANALOGY_BASE_URL` 暂沿用 MLEvolve 的 `gpt-5.6-sol` / `http://cliproxy:8317/v1`，按实际接口修改。

同一对配置重复实验时，结束旧 Job 后重建即可；保留 `RUN_TAG` 的 `$(POD_UID)` 和 worktree 路径的 `$(RUN_TAG)`，无需手动换运行目录。如果并行启动另一对实验，再同步修改 Job 名、pair label、`RUN_TAG` 的 pair 前缀及本组 home 的 `subPath`。主仓库、共享 venv 和 Python 的 `subPath` 保持不变，不会随新实验重新安装依赖。实验期间不要从 dev Pod 修改主仓库、共享环境、固定数据、语料或 Claude 安装。

## 2. 创建 Job，等待启动检查完成

在本项目目录执行：

```bash
kubectl --context nautilus apply -f k8s/job-jigsaw-baseline.yaml
kubectl --context nautilus apply -f k8s/job-jigsaw-analogy.yaml
kubectl --context nautilus -n ecepxie get pods -l experiment-pair=jubias-pair-001
kubectl --context nautilus -n ecepxie logs -f job/autoresearch-jubias-pair-001-baseline
```

Job 先检查主仓库和 prepared 文件路径，执行 `apt-get update` 并安装 `git curl ca-certificates gcc`，确认主仓库干净并记录 HEAD，然后 `source "$AUTORESEARCH_VENV/bin/activate"`。两组均检查所有直接训练和 analogy 依赖是否已安装、记录版本，并检查 Python 路径及 Torch 2.9.1/cu128；缺失依赖会一次列出，analogy 还检查 API/检索库能否导入。全文解析器的实际运行和带凭据的 preflight 仍由 analogy agent 在新 worktree 中完成。随后打印实际 GPU、计算能力、显存和编译架构，执行小型 FP32 CUDA 矩阵乘法、反向传播和 `torch.cuda.synchronize()`，检查结果与梯度。只有这些真实运算及 Claude 可执行文件检查均通过，才显示 `Setup complete` 和 Pod `Ready`；此时 worktree 尚未创建。缺少环境或检查失败会明确报错退出，不会转为重新安装或 CPU 运行。`apt-get` 的系统工具下载仍保留。

如果集群里已经创建了旧版 Job，修改本地 YAML 不会改变现有 Pod 的挂载和启动流程。当前这些非 suspended Job 应在旧运行结束后重建以使用新模板；新 Pod UID 会自动生成新运行目录，旧产物保留。[Job 调度字段更新规则](https://kubernetes.io/docs/concepts/workloads/controllers/job/#mutable-scheduling-directives)

```bash
kubectl --context nautilus -n ecepxie exec -it job/autoresearch-jubias-pair-001-baseline -- bash
# 另一个终端进入 analogy：
kubectl --context nautilus -n ecepxie exec -it job/autoresearch-jubias-pair-001-analogy -- bash
```

## 3. 直接启动 Claude，提供 prompt

两个 Pod 初始都在共享主仓库 `/workspace/autoresearch`。新 `kubectl exec` shell 不会继承 Job 主进程中 `source` 的环境变化，因此进入后先激活共享 venv。分别查看记录的起始 SHA，确认两组一致，并检查 GPU：

```bash
source "$AUTORESEARCH_VENV/bin/activate"
cat /tmp/autoresearch-start-commit
printf 'Run: %s\nWorktree: %s\n' "$RUN_TAG" "$AUTORESEARCH_REPO_DIR"
"$AUTORESEARCH_VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

Claude 从这里读取本组 program 后创建 worktree。后续每次 shell 调用都必须显式进入 `$AUTORESEARCH_REPO_DIR`，编辑工具使用该目录下的绝对路径；一次 shell 的 `cd` 不会改变 Claude 的启动目录或其他工具的默认目录。所有 Python 命令显式使用 `"$AUTORESEARCH_VENV/bin/python"`，不依赖后续 shell 的 PATH，也不再调用 `uv run`。这些要求已写入两份 program。

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
当前目录是共享主仓库；按 program 先用 git worktree add 创建 run 分支和 $AUTORESEARCH_REPO_DIR。
之后只在本组 worktree 修改和运行；禁止在主仓库 checkout 或编辑。
Job 自启动起限时6小时，不要从现在重新计算6小时。
完全不使用 analogy agent、知识库、类比缓存或报告；不 commit 或 push。
```

analogy prompt：

```text
阅读并执行 program-analogy.md，开始 Jigsaw analogy 实验。
当前目录是共享主仓库；按 program 先用 git worktree add 创建 run 分支和 $AUTORESEARCH_REPO_DIR。
之后只在本组 worktree 修改和运行；禁止在主仓库 checkout 或编辑。
Job 自启动起限时6小时，不要从现在重新计算6小时。
必须在 draft 和每次 improve 前使用 analogy agent，不修改其代码；不 commit 或 push。
```

如需让 agent 按剩余时间规划，可查看各 Job 的实际开始时间，把“该开始时间 + 6 小时”的 UTC 截止时间补进 prompt；无需计时脚本：

```bash
kubectl --context nautilus -n ecepxie get jobs -l experiment-pair=jubias-pair-001 \
  -o custom-columns=NAME:.metadata.name,START:.status.startTime,LIMIT_SECONDS:.spec.activeDeadlineSeconds
```

## 4. 查看结果

Pod 到期后回到 dev Pod，从以下 PVC 目录读取结果（将 `<pod-uid>` 换成本组实际值，或按启动日志中的 `RUN_TAG` 查找）：

```text
/workspace/autoresearch/worktrees/jubias-pair-001-baseline-<pod-uid>/results/jubias-pair-001-baseline-<pod-uid>/
/workspace/autoresearch/worktrees/jubias-pair-001-analogy-<pod-uid>/results/jubias-pair-001-analogy-<pod-uid>/
```

本组修改后的 `train.py` 也保留在对应 worktree。agent 不删除 worktree，也不提交代码；后续 Git 操作由你处理。

两组共用中性 `experiment_artifacts.py` 保存训练前源码并绑定完成后的指标、配置、日志；baseline 不调用 analogy 的任何命令。helper 只负责结果绑定，不负责限时。比较 `source.json.execution_status == "completed"` 且 `completed_at_utc` 早于实际 Job 截止时间的候选；未完成或被中断的候选不计分。没有完成候选就报告无有效结果。

保存 Job 的开始/结束状态、两组起始 Git SHA、prepared ID、GPU/环境/主模型配置、`results.tsv`、最佳候选以及 analogy 的检索记录。一次配对适合先确认流程，之后再用预先约定的种子做重复实验。

确认结果后可删除 Job，PVC 上的文件会保留：

```bash
kubectl --context nautilus -n ecepxie delete job autoresearch-jubias-pair-001-baseline autoresearch-jubias-pair-001-analogy
```
