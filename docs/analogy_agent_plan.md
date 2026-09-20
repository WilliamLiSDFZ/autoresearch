# 在 autoresearch 中接入 analogy agent 的方案

状态：原始方案记录，日期：2026-09-19。用户随后要求首版即具备 MLEvolve 的全文阅读能力；该部分与独立脚本已实现，`program.md` 接入仍待进行。以下“首版摘要、全文后续扩展”等旧范围已被此要求替代；实际接口、依赖和使用方式见 [实现说明](/Users/william/Documents/project/python/autoresearch/docs/analogy_agent.md)。

## 1. 结论与范围

可行。建议在 autoresearch 中增加一个独立的 `analogy_agent.py`，由运行实验的 Claude Code 按 `program.md` 在 **draft 前**和**每轮 improve 前**调用。脚本读取 Agentic Knowledge Base 的论文库，通过 LLM 驱动检索与类比分析，输出有证据的实验建议；Claude Code 根据建议修改 `train.py`。

保留 autoresearch 现有的实验循环、五列 `results.tsv`、固定 Jigsaw 数据与评测。`program.md` 只做局部增补，不整体重写。暂不添加单次训练、检索或整个 pipeline 的运行时间上限。检索代码、提示词和检索配置在实验期间冻结，明确禁止实验 agent 修改。

本次只编写此方案，不实现脚本、不修改现有运行代码、不运行实验、不操作 Git。后续分支与提交由用户管理。

## 2. 两个项目目前实际提供了什么

| 部分 | 读到的实际实现 | 对接方式 |
| --- | --- | --- |
| Agentic Knowledge Base | 本地 `paper_corpus` 的 manifest 记录 12,800 篇论文，schema 2；JSONL 包含标题、TLDR、摘要、论文 ID、来源及 PDF URL | 作为外部只读语料使用，不复制整个知识库进 autoresearch |
| BM25 检索 | 索引文本为 `title + title + tldr + abstract`，分词包含停用词过滤与 Porter stemming；搜索先返回紧凑元信息，摘要另外读取 | 移植小范围检索逻辑，固定分词方式和依赖版本 |
| MLEvolve draft | 首个生成草稿前调用，输入任务、数据描述与资源信息；没有已执行候选的结果 | 对应首次设计 Jigsaw 方案、编写任务实现之前 |
| MLEvolve improve | 在父候选基础上规划与改代码之前调用，结合代码、验证结果和尝试历史 | 对应 autoresearch 每次从当前保留版本提出下一轮改进之前 |
| 类比报告 | 说明问题结构、来源机制、如何映射到当前任务、假设、约束和验证办法 | Claude Code 每轮至多选择一个机制，也可以明确拒绝全部建议 |

这里应以实际源码为准：知识库 README 中较早的向量检索说明已经过时，相关设计文档明确标记该方案已被替换。当前不是接入一个 FAISS 服务，也不需要重新生成 embedding。

MLEvolve 默认配置是 `enabled=false`、`draft=false`、`improve=true`，因此“代码支持两个阶段”不等于默认会在两个阶段都调用。其当前 context v2 已实现；全文读取也已实现，但默认关闭。默认检索参数是 14 轮、每次 top-k 10、最多 3 个机制、报告上限 12,000 字符。这些可以作为首版起点，属于交互次数和上下文大小限制，不是实验运行时间限制。

另外，现有 draft 主要注入 Markdown 报告；improve 才有更完整的结构化选择与代码交接记录。本方案统一两阶段的报告和选择记录，是移植时的增补。

来源：[语料 manifest](/Users/william/Documents/project/python/Agentic_Knowledge_Base/output/paper_corpus/manifest.json:1)、[BM25 实现](/Users/william/Documents/project/python/MLEvolve/engine/analogy/corpus.py:68)、[旧方案说明](/Users/william/Documents/project/python/Agentic_Knowledge_Base/docs/semantic_retrieval_design.md:1)、[draft 接入](/Users/william/Documents/project/python/MLEvolve/agents/draft_agent.py:29)、[improve 接入](/Users/william/Documents/project/python/MLEvolve/agents/improve_agent.py:31)、[默认配置](/Users/william/Documents/project/python/MLEvolve/config/config.yaml:36)。

## 3. 独立脚本的职责

建议区分三个角色：

1. **Claude Code 主 agent**：决定下一步实验，调用检索脚本，阅读报告，选择或拒绝建议，修改并运行 `train.py`。
2. **`analogy_agent.py`**：内部调用配置好的 LLM，让它分析问题、选择检索词、阅读论文证据、提出类比机制；不修改或执行候选训练代码。
3. **论文库**：只提供可检索的数据，不承担模型推理或执行实验。

内部流程为：

```text
阶段输入 → 区分事实、假设和未知项 → 提炼问题结构
    → 生成机制相关检索词 → BM25 搜索 → 阅读候选摘要
    → 验证证据与任务映射 → 输出 report.json / report.md
    → Claude Code 选择至多一个机制或拒绝 → 进行实验
```

例如，输入可以提示“不同子群的排序表现存在差异”，再检索与群体风险、样本权重或排序目标有关的机制。不能仅搜索任务名称后罗列相似论文，也不能在缺少诊断证据时把某个问题断言为性能瓶颈。

首版保留一个公开 Python 入口，将 BM25、输入整理、工具循环、报告验证和输出逻辑放在其中；从 MLEvolve 移植必要的小模块逻辑并记录来源版本。不要直接 `import MLEvolve` 或调用依赖 `SearchNode` 的接口，否则会连带引入搜索树、运行状态和配置系统。

### 模型与依赖

独立 agent 需要单独配置模型 ID、API endpoint 和凭据。不能假设已经登录 Claude Code，就自动具备可供 Python 脚本使用的 API 凭据。首版可沿用 MLEvolve 所用的 OpenAI SDK/API 适配方式，明确支持的接口类型；凭据通过环境变量提供，不写入报告。

当前 autoresearch 的依赖中没有 `rank-bm25`、`nltk`、`openai` 或 `jsonschema`。实现阶段需要提前补充实际用到的依赖并锁定版本；两组使用相同环境。实验开始后继续遵守“不临时安装依赖”的规则。固定使用 Porter stemming，缺少依赖时报错，避免静默切换分词算法。

如果只让 Python 返回 BM25 结果、把全部类比推理交给外层 Claude，也能工作，但那是另一种架构。本方案采用“Python 内部有独立 LLM 检索循环”，更接近 MLEvolve 的 analogy agent。

### 摘要与全文

首版范围为 BM25、摘要阅读、结构化类比报告和两个阶段的调用。每条建议至少需要来自本次实际读取摘要的证据，并标记为 `abstract_only`；摘要可能已被语料构建器截断，不应推断论文未提供的实现细节。

这是比 MLEvolve 默认摘要模式更严格的证据要求：当前源码在全文工具关闭时主要检查论文 ID 曾被搜索返回，并不强制读过摘要。

全文能力可作为后续独立扩展：迁移 `open_paper/read_paper`、解析 worker、缓存与证据校验，而不只是下载 PDF。若启用，需在一组对照开始前固定配置；无法读取时明确标注摘要回退。摘要版不能被描述为完整复现启用了全文的 MLEvolve 配置。

来源：[检索循环入口](/Users/william/Documents/project/python/MLEvolve/engine/analogy/agent.py:700)、[报告证据校验](/Users/william/Documents/project/python/MLEvolve/engine/analogy/report_v2.py:135)、[全文读取设计](/Users/william/Documents/project/python/Agentic_Knowledge_Base/docs/analogy_fulltext_reading.md:37)。

## 4. draft 与 improve 分别何时调用

autoresearch 没有显式的 draft node 和 improve node，需要在现有顺序循环中定义对应时刻。

| 阶段 | 调用时机 | 可用输入 | 输出如何使用 |
| --- | --- | --- | --- |
| draft | 当前 run 尚未形成首个任务方案，在编写首个 Jigsaw 实现之前调用一次 | 任务描述、公开数据的 schema/统计摘要、固定指标定义、资源和允许的模型/依赖 | 辅助首次方案设计，再建立该 run 的初始分数 |
| improve | 每轮改进开始，在修改当前保留且成功运行的版本之前调用 | 父实验的源码快照、配置、公开验证指标、运行摘要、该 run 已尝试的改动 | 辅助选择本轮一个可验证的改动 |
| debug | 修复同一候选的报错 | 原有检索报告、错误日志 | 不另开一次类比检索；继续修复或按原规则放弃 |

`draft` 不接收另一组的实验结果，也不把未执行的计划写成观察事实。`improve` 必须使用当前保留父实验的信息；上一轮如果被 discard，下一轮不能把它误当成当前实现，但可以把失败尝试放入历史。

对 Jigsaw，improve 应读取固定评测产生的 `score`、`overall_auc`、三类子群 AUC 的聚合值，以及可用的逐身份组指标和样本计数。只看总分会丢失有用的诊断信息。报告需区分“某项指标较低”的观察与“某训练机制导致该现象”的待验证假设。

### 保证代码与运行结果对应

当前 `train.py` 已保存 `metrics.json`、`config.json`、预测和 checkpoint，但没有在实验目录保存执行时的源码快照。实现接入时，应在每次启动训练前由外层流程保存 `train.py` 快照及 SHA256，运行后归档对应日志和结果索引；不需要为此改变训练算法。

improve 只读取这份已完成父实验的快照，不能把工作区刚改过、尚未执行的 `train.py` 配上旧指标。缺少可靠对应关系时标记为未知，不伪造。读取代码用文本/AST 和行号引用，不 import 或执行训练脚本；无需打开 checkpoint。

来源：[当前训练输出](/Users/william/Documents/project/python/autoresearch/train.py:575)、[MLEvolve 只读代码工具](/Users/william/Documents/project/python/MLEvolve/engine/analogy/code_tools.py:122)。

## 5. 建议的命令与产物

以下是拟实现的 CLI，当前尚不可运行。路径以 Nautilus 的持久化目录为例，任务说明文件需使用实际存在的路径；`--prepared-dir` 必须与训练所用 `AUTORESEARCH_JIGSAW_DIR` 一致。示例中的占位符需替换。

```bash
# 初始设计前，只调用一次
uv run /workspace/autoresearch/analogy_agent.py \
  --stage draft \
  --task-file <任务说明文件的绝对路径> \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --corpus-dir /workspace/Agentic_Knowledge_Base/output/paper_corpus \
  --output-dir /workspace/autoresearch/results/<run>/analogy/draft-000

# 每轮改进前，输入当前保留的父实验
uv run /workspace/autoresearch/analogy_agent.py \
  --stage improve \
  --task-file <任务说明文件的绝对路径> \
  --prepared-dir /workspace/autoresearch/results/jigsaw-data \
  --corpus-dir /workspace/Agentic_Knowledge_Base/output/paper_corpus \
  --parent-artifacts /workspace/autoresearch/results/<run>/<parent-experiment> \
  --history /workspace/autoresearch/results.tsv \
  --output-dir /workspace/autoresearch/results/<run>/analogy/improve-001
```

脚本从这些输入构建受限上下文，不扫描整个工作区。只读取公开任务元信息、冻结代码和必要结果，不向模型发送完整训练集、私有标签或凭据。论文和日志内容作为资料处理，不能改变工具权限；检索 agent 没有 shell 或代码写入工具。

每次调用建立独立目录，不覆盖已有记录：

| 文件 | 用途 |
| --- | --- |
| `context.json` | 实际发送的阶段输入，区分观察、假设、未知项 |
| `trace.jsonl` | 工具调用、检索词、返回的论文 ID/排序、实际交付的证据文本；不要求记录模型私有推理 |
| `report.json` | 经 schema 和证据检查的结构化报告，是唯一报告数据源 |
| `report.md` | 从同一 JSON 渲染，供 Claude Code 阅读 |
| `manifest.json` | run/stage/call ID、父实验、prepared ID、源码/脚本/语料 hash、模型与非敏感配置、调用状态、token 和耗时 |
| `decision.json` | 主 agent 记录所选机制 ID 或 null、理由、具体适配及后续实验 ID |

在该 run 的 analogy 目录另存 `index.jsonl`，关联报告、决策、父子源码 hash、实验结果及 keep/discard/crash。保留原五列 `results.tsv`，不把大段检索内容塞进表格。父实验目录与本次 run、prepared ID 的匹配应校验；显式登记的共同初始候选允许跨 run 引用。

报告最多提出 3 个机制，每个必须包含：目标问题及依据、论文 ID 和已读证据、来源与目标的结构映射、建议修改的位置与动作、适用假设、必须保留的约束、预期观察及拒绝条件。主 agent 每个实验至多采用一个，也可以拒绝全部建议；采用时保留完整机制和约束，不能只截取建议标题。

improve 中对现有实现的判断使用 `code_refs={experiment_id, source_sha256, start_line, end_line}`；运行事实使用 `runtime_evidence` 指向公开指标或日志中的具体字段/位置。校验引用必须落在本次实际交付给模型的代码行或运行证据内，不能仅凭文件存在就视为已阅读。

“选择了机制”“代码实际实现了它”“分数提高了”分别记录，不能从声明或分数提升直接推出类比产生了因果收益。

## 6. 对 program.md 的最小改动

保留现有五个章节及循环顺序，仅在以下位置插入规则：

1. **Setup**：按分支末尾 `baseline` / `analogy` 确定实验类型；analogy 组预检模型配置、依赖、语料和冻结版本。两组都保存代码/结果来源记录，baseline 不调用 analogy 脚本或读取 analogy 报告。
2. **What you CANNOT do**：增加禁止修改 `analogy_agent.py`、内部提示词、工具/schema、固定检索参数及论文库；`prepare.py` 和固定评测的原有限制继续保留。研究代码仍只允许修改 `train.py`，允许写入日志、报告和决策等运行产物。
3. **The first run**：对于需要创建初始任务实现的 analogy run，在首次方案设计前执行 draft 检索、读报告并记录选择，再编写和运行初始候选。已有共同初始候选时按第 8 节的 improve-only 模式处理。
4. **The experiment loop 第 2 步**：analogy 组先对当前保留父实验执行 improve 检索、阅读报告和记录选择，再修改 `train.py`。debug 沿用本次报告。
5. **运行和记录步骤**：补充执行前源码快照、运行后结果归档，以及报告与实验的关联；评分、keep/discard 和五列 TSV 规则不变。

可加入的关键英文约束示意：

> On analogy runs, invoke the fixed analogy agent before the initial draft and before each improvement of the current kept experiment. Read its report before editing train.py. Select at most one mechanism, or record why all suggestions are rejected. Do not modify analogy_agent.py, its prompts, retrieval settings, tools, report schema, or the paper corpus. Baseline runs must not invoke the analogy agent or consume its reports.

这段只是插入内容示意，具体命令和失败处理将在实施时补齐。不要把本方案全文复制进 `program.md`。

## 7. 冻结、部署与异常行为

### 冻结与持久化

- 开始运行前固定检索脚本、提示词、schema、模型/生成配置、依赖与分词版本、语料实际内容、prepared ID 和评测版本。运行中不自动重建语料或调整检索规则。
- 核对 JSONL 的条数、schema、重复 ID，计算实际 SHA1 前 12 位与现有 manifest 对照，并另外记录完整 SHA256。MLEvolve 当前只是读取 manifest 的 digest，不能将其误称为已验证的内容 hash。
- `program.md` 的禁止修改是行为约束；外层在调用前后核对固定文件 hash 可发现意外修改。需要更强约束时，在 Nautilus 将检索脚本/配置和语料单独只读挂载，训练代码与结果目录保持可写。不要把脚本内部自检称为强制隔离。
- 脚本、语料、结果必须位于实际挂载的 PVC 路径上；仅有 `/workspace` 这个路径名不保证持久化。API 凭据通过运行环境注入。
- MLEvolve 的 BM25 缓存只在当前 Python 进程内有效。按阶段启动独立 CLI 时会重新建索引；首版接受这个开销并记录，后续需要磁盘缓存时按语料、分词器和版本共同失效。

### 状态与失败处理

| 情况 | 行为 |
| --- | --- |
| baseline | 记录检索禁用，不调用检索模型 |
| analogy 启动时缺依赖、配置、语料，或冻结 hash 不匹配 | 预检失败，修复启动配置后再开始；不得静默当成正常 analogy run |
| 检索完成且有合格建议 | 状态 `ok`；主 agent 可采用或拒绝 |
| 正常检索后没有可信、适用的建议 | 状态 `abstained`，保存理由；主 agent 按原流程提出实验 |
| API/工具错误，或交互轮数耗尽仍无有效报告 | 状态 `failed` 并保存具体原因；主 agent 可继续原流程，但本次记为检索失败，不能伪装成成功检索或有效 abstention |

失败后不得修改冻结脚本来“修好这一轮”。应记录问题；确需修复检索实现时，作为用户管理的独立版本更新，并在新协议版本下运行。汇总对照结果时同时报告检索成功率、拒绝率和失败调用，不能隐藏退化调用或把该 run 改标 baseline。

## 8. 与现有 baseline 如何做对照

当前本地 [run.log](/Users/william/Documents/project/python/autoresearch/run.log:554) 记录了一次完成的训练：`val_score=0.921968`，overall AUC 为 `0.962882`。这是日志结果；本次没有重新评测远端预测或 checkpoint，也不能据此认定整个自主实验循环已经结束。

接入支持两个阶段，但比较对象需要明确：

| 跑法 | 两组的共同起点 | 能回答的问题 |
| --- | --- | --- |
| **完整 draft + improve 对照（推荐用于原设想）** | 同一任务模板、数据、评测和环境；两组分别形成初始实现，analogy 组在编写前调用 draft | 类比检索加入整个研究过程后的效果 |
| **从共同 baseline 改进** | 同一份已完成训练的初始候选代码、配置和公开结果 | 类比检索对后续 improve 的作用 |

第二种可以直接利用现有 baseline，但不应把对已有实现的第一次检索叫 draft，也不能用它单独证明 draft 检索有效。第一种的 analogy draft 不应读取 baseline 组已经得到的验证结果或优化历史。

目前任务分支 `codex/jigsaw-unintended-bias` 指向 `b900b05`，Jigsaw 训练实现由后续 `c85845b` 引入。完整对照可在接入实现准备好后，由用户确定并冻结双方共同的任务起点；不能把 baseline 分支中已经适配完成的 `train.py` 自动当成“设计前”状态。现有 baseline 可以保留为参考；是否纳入严格配对，要核对其实际起点、模型配置和执行条件。

分支命名继续使用 `run/<YYYYMMDD_HHMMSS>-jigsaw-unintended-bias-in-toxicity-classification-<baseline|analogy>`。两组固定相同的数据/评测、GPU、可用模型与依赖、主 agent 配置和实验记录方式；区别是是否启用类比检索。不同 run 的 TSV 和上下文需隔离，不能把别组历史混入本组输入。

暂不设置运行时间限制。保存每次训练与检索的实际耗时、token、已完成候选数，并比较分数随实验次数及实际耗时的变化；这些是观测，不是截止条件。单次最高分不足以判断稳定收益，后续可在相同设置下重复独立 runs。检索增益也包含额外模型调用带来的作用；如果以后要区分“知识库证据”与“额外推理”的贡献，需要另加相同调用预算但不提供检索证据的消融组，本次先不扩展范围。

## 9. 后续实施顺序与验收

1. **实现独立检索入口**：移植 BM25 与必要 agent 循环；加入模型配置、摘要证据验证、结构化报告及轨迹输出。记录移植来源，使用小语料和模拟模型响应验证流程。
2. **整理阶段上下文与运行记录**：补齐源码快照和父实验关联；draft 只用任务输入，improve 只用有来源的代码/公开结果。首版不依赖完整搜索树或论文全文解析服务。
3. **小幅修改 program.md**：加入上述调用点和不可修改条款，保留原结构及实验循环。正式对照前完成依赖准备与冻结，由用户操作 Git。
4. **验证接入**：先验证 baseline 不调用检索、两个阶段能输出报告、无建议/失败均有明确状态、错误父实验/语料或引用被拒绝、固定文件未改变；再在 Nautilus 做一次真实检索试跑，最后开始 GPU 实验。

验收重点：检索确实发生在写代码之前；每个机制都有本次读过的论文证据；报告中的已知事实可追溯到任务或已执行父实验；只有 `train.py` 的研究实现可变；固定数据与指标不受影响；原始检索、采纳决策和实验结果可以对应起来。当前文档阶段不进行付费模型调用或训练验证。
