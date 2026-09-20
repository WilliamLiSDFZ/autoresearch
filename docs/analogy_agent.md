# 独立 analogy agent 使用说明

`analogy_agent.py` 已实现 MLEvolve 风格的 draft / improve 检索，包括论文全文阅读。`program-analogy.md` 要求在这两个阶段调用；`program.md` 是禁止使用它的 baseline 协议。Nautilus 配对实验请先看 [运行说明](jigsaw_comparison.md)，其中两组统一使用中性 `experiment_artifacts.py` 归档。

## 能力与边界

主入口下的 `autoresearch_analogy/` 模块来自 MLEvolve 的实际实现：BM25、原始两阶段提示词、context v2、论文工具、代码工具、报告验证/修正和 Responses transport。运行时不 import MLEvolve，也不需要复制整个知识库。

| 能力 | 当前实现 |
| --- | --- |
| 论文检索 | BM25、标题加权、停用词、Porter stemming；搜索后单独读摘要 |
| 全文阅读 | 默认开启 `open_paper` / `read_paper`；目录分页，选择方法、实验、限制、附录对应片段 |
| PDF 解析 | 独立 CPU 子进程，PyMuPDF / PyMuPDF4LLM，正文/表格文本；不执行论文中的代码 |
| 缓存 | 绑定论文、来源 URL、reader 代码和解析器版本，检查 PDF / 文本 hash；可离线读取已有兼容缓存 |
| 论文证据 | 只接受本轮实际读过的摘要或正文引文；打开目录不等于读过正文；保留页码、chunk 和 hash |
| improve 实现证据 | 已完成父实验的源码索引、按行读取、显式提供历史候选时可比较 diff；只读 allow-list，不执行候选代码 |
| 类比报告 | 结构映射、实际观察、假设、未知项、目标适配、约束、验证与拒绝条件；保留完整机制 |
| 报告修正 | 共 14 轮，通常第 12 轮前首次提交，剩余轮修复引用/schema；修正期间限制扩展搜索 |
| 模型接口 | OpenAI-compatible Chat Completions 与 Responses；后者完整重放 tool calls 和 opaque reasoning 状态 |

MLEvolve 默认关闭全文；此版本按需求默认开启，并要求 context v2。保留其每轮最多打开 3 篇论文、12 次正文读取、每次 8,000 字符、合计 40,000 字符等默认值。全文无法取得时可以根据已读摘要给出明确标注的建议，也可以 abstain，不能假装读过全文。

保留下载/解析的单次 I/O 超时及累计打开预算，用于处理挂起的服务或解析进程；它们不限制训练或整个研究循环。CLI 没有实验 deadline、watchdog 或训练时长参数。

## 安装与配置

在 Nautilus 的 autoresearch 目录中使用原来的 `uv sync --locked`，默认会安装新增的 `analogy` 依赖组。只想在 CPU/Mac 上运行检索或测试时，可以单独安装它，避开训练用的 CUDA PyTorch：

```bash
uv sync --locked --only-group analogy --python 3.10
```

全文相关版本与 MLEvolve 对齐：PyMuPDF / PyMuPDF4LLM / PyMuPDF Layout 1.28.0，OpenAI SDK 1.66.3，rank-bm25 0.2.2，NLTK 3.9.1。额外将 Python 3.10 的 ONNX Runtime 固定为 1.23.2，避免依赖解析选到缺少 CPython 3.10 wheel 的版本。Porter stemmer 不需要下载 NLTK 语料包。

通过运行环境设置以下变量。不要把真实凭据写入命令示例、Git 或 JSON 配置：

- `ANALOGY_MODEL`：实际可用的模型 ID，必填。
- `ANALOGY_API_KEY`：API 凭据，也支持 `OPENAI_API_KEY`。
- `ANALOGY_BASE_URL`：兼容 endpoint；可省略，或使用 `OPENAI_BASE_URL`。

`--model`、`--base-url` 可以覆盖对应非敏感环境设置。`--api auto` 沿用 MLEvolve 的模型路由；可显式指定 `responses` 或 `chat_completions`。不要假定 Claude Code 登录信息能作为此脚本的 API 凭据。

Responses 的 `--reasoning-effort` 默认 `high`。Chat 路径沿用 MLEvolve 的模型兼容规则：部分 OpenAI reasoning 模型的工具调用实际用 `none`，其他模型不传 reasoning 参数；运行记录分别保存 requested / effective 值。Claude 模型需由支持其工具调用的 OpenAI-compatible endpoint 提供，本版本不直接实现 Anthropic 原生 API。

高级设置可通过 `--config /absolute/path/analogy-config.json` 提供。仅接受 `agent`、`context`、`code_tools`、`fulltext` 四个对象，未知字段或非法值会报错。例如：

```json
{
  "agent": {"max_turns": 14, "top_k": 10, "max_mechanisms": 3, "report_char_budget": 12000},
  "fulltext": {"enabled": true, "max_papers": 3, "max_read_calls": 12, "read_chars": 8000, "total_chars": 40000}
}
```

不设置时使用移植后的默认值。根据实际 endpoint 能力设置 context 上限；默认上限来自 MLEvolve 的配置，不是脚本对服务端能力的测量。

## draft：初始方案之前

以下示例使用 Nautilus 路径，需要将任务说明与 run 名替换为实际值。检索只读已准备数据的 manifest，不读取整份数据集；开始实验前仍应使用 `prepare.py verify` 验证实际数据文件。

```bash
uv run /workspace/autoresearch/analogy_agent.py preflight \
  --stage draft \
  --task-file /workspace/data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --corpus-dir /workspace/Agentic_Knowledge_Base/output/paper_corpus \
  --lock-file /workspace/autoresearch/results/my-analogy-run/analogy-protocol.json

uv run /workspace/autoresearch/analogy_agent.py run \
  --stage draft \
  --task-file /workspace/data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --corpus-dir /workspace/Agentic_Knowledge_Base/output/paper_corpus \
  --lock-file /workspace/autoresearch/results/my-analogy-run/analogy-protocol.json \
  --output-dir /workspace/autoresearch/results/my-analogy-run/analogy/draft-000
```

`preflight` 校验本地输入、依赖和模型配置是否齐备，不调用模型、不确认凭据有效性、不下载 PDF。`--lock-file` 创建固定协议记录，已存在则拒绝覆盖。后续 `run` 指定同一文件会核对代码、配置、依赖、任务、prepared ID 与语料版本；调用后再次检查。对照实验建议始终使用它。

draft 明确拒绝父实验、历史结果和参考候选输入。输出后阅读 `report.md`，每次至多选择一个机制，或者说明拒绝原因。选择与训练由外层 agent 完成，脚本不会自动编辑或运行 `train.py`。

## improve：已完成候选之后

为避免“新代码配旧结果”，父实验需要源码与结果对应的记录。训练脚本必须保存对应配置和完整评测指标；当前任务模板仍需在 draft 中适配 Jigsaw。下面是独立调用 analogy CLI 的归档示例，假设训练脚本按 `RUN_TAG` / `EXPERIMENT_ID` 写入同一目录；配对实验应按 program 使用中性 `experiment_artifacts.py` 和 `AUTORESEARCH_ARTIFACT_DIR`，baseline 禁止调用这里的命令：

```bash
# 训练前：新实验目录必须不存在。
uv run /workspace/autoresearch/analogy_agent.py snapshot \
  --source /workspace/autoresearch/train.py \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --artifact-dir /workspace/autoresearch/results/my-analogy-run/exp000 \
  --run-id my-analogy-run --experiment-id exp000

# 显式启动训练；RUN_TAG / EXPERIMENT_ID 必须指向刚才的目录。
RUN_TAG=my-analogy-run EXPERIMENT_ID=exp000 \
  uv run /workspace/autoresearch/train.py \
  > /workspace/autoresearch/results/my-analogy-run/exp000/run.log 2>&1

# 确认训练成功后、再次改代码前，将配置和指标与快照绑定。
uv run /workspace/autoresearch/analogy_agent.py complete \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --artifact-dir /workspace/autoresearch/results/my-analogy-run/exp000

uv run /workspace/autoresearch/analogy_agent.py run \
  --stage improve \
  --task-file /workspace/data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/description.md \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --corpus-dir /workspace/Agentic_Knowledge_Base/output/paper_corpus \
  --parent-artifacts /workspace/autoresearch/results/my-analogy-run/exp000 \
  --lock-file /workspace/autoresearch/results/my-analogy-run/analogy-protocol.json \
  --output-dir /workspace/autoresearch/results/my-analogy-run/analogy/improve-001
```

`complete` 不重新训练或评测，只校验源码未变、结果字段与 prepared ID/指标一致，然后固定文件 hash。训练报错或没有合法最终指标时不要执行 complete。它提供可核对的记录，不是对训练执行真实性的独立证明。

后续 improve 选择当前 keep 的父实验。可加 `--history /workspace/autoresearch/results.tsv` 提供本 run 的五列历史表；TSV 本身没有 run ID，调用者需要确保没有混入另一组历史。可重复传入 `--reference-artifacts /absolute/path/to/completed-experiment`，允许代码工具读取/比较同 run、同 prepared ID 的已完成历史候选。不会自动扫描其他分支或目录。

已跑完的旧 baseline 没有上述执行前快照时，脚本不会猜测其代码/指标对应关系。首次验证 improve 可从明确保存快照的新候选开始；旧产物只有在核对确切执行源码和结果后，才适合迁移成同样的记录。现有 baseline 文件不会被脚本修改。

## 输出、缓存与失败

每次调用使用新的 `--output-dir`，已有目录不会被覆盖，包含：

- `report.json` / `report.md`：同一份验证后的报告；机制包含稳定 ID。
- `context.json`：实际输入与可见的运行事实；`context.json` 不会被后续的上下文预算 sidecar 覆盖。
- `context_budget.json`：各轮上下文估算及提交预留信息。
- `trace.jsonl`：模型可见的文字、工具调用、检索结果和返回证据。
- `fulltext.json`：打开记录、实际交付片段、摘要和来源 hash。
- `code_reads.json`：代码读取和行号引用记录。
- `model_calls.json` / `submission_attempts.json`：调用用量及报告修正过程；不保存 opaque reasoning 内容。
- `manifest.json`：状态、耗时、父实验来源和冻结配置。

状态为 `ok`、`abstained` 或 `failed`；另外保存 MLEvolve 的 `accepted_complete` / `accepted_partial` 等交付状态。正常报告或明确 abstention 返回 exit code 0，检索失败返回 1，启动/配置/输入或内部错误返回 2。不能把 API 错误记成“没有合适论文”。部分机制因证据不足被剔除时，详细原因保留在 submission attempts。

默认全文缓存是 `/workspace/autoresearch/results/analogy-cache/paper_fulltext`（由项目所在位置推导），可用 `--cache-dir` 覆盖。`--offline` 仅禁止 PDF 网络下载，模型 API 仍会联网。不同 reader/解析器版本的缓存不会混用；因此不会直接认领 MLEvolve 原缓存为兼容缓存。

将项目、语料与结果放在实际挂载的 PVC 上。`/workspace` 名字本身不意味着持久化。实验时固定入口、整个 `autoresearch_analogy/`、配置及语料；freeze hash 是检测机制，更强约束可用独立只读挂载。当前没有修改 `program.md`，所以自动调用和外层采纳决策记录要等下一步接入。

## 验证与移植来源

本地离线测试：

```bash
uv run --locked --only-group analogy python -m unittest discover -s /workspace/autoresearch/tests -v
```

覆盖语料校验、真实 PDF 子进程解析、缓存/offline、代码和论文证据、报告修正、Responses 重放/错误处理及 CLI 的文件绑定与冻结。使用模拟模型和临时 PDF，不需要 GPU 或付费 API。未将模拟测试当成真实模型检索质量评估。

移植参考的 MLEvolve HEAD 为 `8166025d6e465d558bd1297ccb92bd8391781d2a`，知识库 HEAD 为 `cb8eae3e0d2ca86637e6abb47f53ab682e300c94`；以此次读取的工作区源码为依据。主要对应 `engine/analogy/` 及 `llm/responses.py` / `model_profiles.py`。与搜索树耦合的 node 调度改成显式 CLI 输入；核心阅读、证据和报告逻辑保留。新增了语料实际 hash 校验、明确失败状态、阶段输入检查与冻结记录。
