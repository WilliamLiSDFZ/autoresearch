# Autoresearch Vendi 分析

**完整方案分析**使用 `compare_solution_vendi.py`，不需要父节点，输出到独立的 `results/vendi-solutions/`；见 [完整方案 Vendi 说明](solution_vendi.md)。下文保留原来的变更分析口径，两者不能混算。

`compare_vendi.py` 比较候选实现的机制多样性。它沿用 Agentic Knowledge Base 的 `diff-v1` 证据评估、`vendi-change-packet-v2` 上下文构造和 q=1 cosine Vendi 算法；这些模块随仓库保存，来源 SHA-256 位于文件头，无需旁边存在 AKB 仓库。

默认读取 `~/nautilus/autoresearch-result`，输出到本仓库 `results/vendi/`，缓存到 `results/vendi-cache/`。不连接 Kubernetes，不执行候选代码或更改实验文件。

## 先检查，再提取

```bash
# 仅标准库；不写文件、不调用 API、不下载模型
python3 compare_vendi.py --dry-run

# 保存 parent_map.csv、coverage.csv、samples.jsonl 和 REPORT.md；仍不调用模型
python3 compare_vendi.py --prepare-only

# 自动使用已有父节点证据；缺失关系保留在 coverage 中
# 使用独立脚本环境，不改训练依赖或 uv.lock
uv run --script compare_vendi.py --stages improve
```

`--parent-map` 是可选的人工补充输入，通常不需要传。**不要把自动导出的 `results/vendi/parent_map.csv` 原样传回去**：它是包含待确认行的清点表，不是已审核的输入。没有 baseline 的足够父节点证据时，上面的命令只能分析可用候选，不能生成完整 paired/effect 对照。

正常模式会将候选源码 diff/相关上下文发送到配置的分析 LLM；首次 embedding 可能下载模型。配置优先级：

| 配置 | 来源 |
|---|---|
| API key | `VENDI_API_KEY` → `ANALOGY_API_KEY` → `OPENAI_API_KEY` → `LLM_API_KEY` |
| Endpoint | `--base-url` → `VENDI_BASE_URL` → `ANALOGY_BASE_URL` → `OPENAI_BASE_URL` → `LLM_BASE_URL`；未提供时用 OpenAI 默认地址 |
| 分析模型/API | `--summary-model gpt-5.6-terra`，`--summary-api responses`；也支持 `chat` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`，固定 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，CPU，512 tokens |

密钥由环境传入，不读取其他项目的 `.env` 或实验日志寻找密钥。已有包含 NumPy、Matplotlib、OpenAI 和 sentence-transformers 的分析环境也可直接运行 Python。离线检查通过不代表配置的远端 provider 一定支持所选 API；正式提取保留每条失败记录。

## 父节点证据

初始 draft 描述完整实现；improve 只描述真实 parent→child 的变化。候选源码必须与 `source.json` 的 SHA-256、run/trial 和 prepared ID 一致。读取器接受显式 source receipt 的 ancestry；analogy 记录则交叉检查 `adoption.json`、调用 `context.json`、`manifest.json` 与父源码 hash。

没有可靠关系时保留候选并记录缺失，不使用前一个 trial、最近创建时间或源码相似度猜 parent。Baseline 的首次 draft 可由第一条候选及 TSV 中明确的 initial/draft 描述识别；没有 stage 证据的后续候选显示为空 stage，仍计入覆盖率。

导出的 `parent_map.csv` 可作审阅模板。**复制到另一个文件再编辑**：正式输出会更新，不能把唯一的人工记录放在自动生成路径。只保留已审定的行，未确认行可以从映射文件移除；原候选仍保留为缺失。

`reviewed-parents.csv` 是下例中自定的文件名，不会自动创建。仅在你已整理出有证据的映射后使用：

```bash
uv run --script compare_vendi.py --parent-map results/reviewed-parents.csv --stages improve
```

若报 `missing or invalid stage, evidence`，错误中的 CSV 行号和 run/trial 标识指向未确认记录；单纯改文件名或补上 `improve` 不能证明父节点。可以去掉 `--parent-map` 使用自动证据，或在独立副本中补齐真实依据并移除未确认行。

```csv
run,candidate_id,stage,parent_id,child_sha256,parent_sha256,evidence
actual-run-directory,trial0002,improve,trial0001,<完整child SHA256>,<完整parent SHA256>,具体日志/源码依据或人工审阅记录
```

`draft` 行的 parent 和 parent hash 必须为空；`improve` 必须提供同 run 内父 ID、双端 hash 和证据说明。已有可信记录不能被 CSV 覆盖成另一父节点。未知 ID、重复行、错误 hash、自父、环和父节点创建晚于子节点都会拒绝。人工映射是带依据的审阅断言，脚本验证身份/hash/结构，不声称自动证明其谱系真实性。

## 提取与统计

- 主分析保留失败/pending 的可信源码。`is_valid` 单独由 completed 回执及指标绑定验证；`--valid-only` 是另存目录的敏感性分析，不能把原先失败的候选默认为成功。
- Improve 向模型提供完整 diff、相关源码与静态调用/使用位置。过大的完整 diff 标为证据不足，不静默截断。分为 `changed`、`no_change`、`insufficient_evidence` 三态；仅 changed 的中性描述进入 embedding。
- 摘要排除 arm、指标分数、论文名及类比叙述，控制总字数。保留实际改动类型，包括参数变化、bug 修复和重构。静态连接不证明运行时生效或科学创新。
- `no_change`、证据不足、缺 parent 和提取失败都进入 coverage，不补零、不把拒绝文本嵌入。不同候选重复同一机制仍是不同观察；重复导出同一候选去重，冲突重复报错。
- 每个固定的 study/task/source-protocol/stage/view 分组内，至少有 2 个可用候选的 run 形成固定集合，共用 `m=2..最小候选数`。组合数不超过 1,000 时穷举，否则采样 1,000 次；单次抽样不放回，不同抽样可重复，`--seed` 固定随机性。
- 每个候选数 m 计算各 run Vendi、arm 描述性均值、显式配对差值及配对均值。不同任务、study 或阶段不混合；混合提取版本/模型或 embedding 配置会拒绝。
- Effect = treatment Vendi − baseline Vendi；正值表示更多样，不表示更好。图中的区间是候选子集变化范围，不是实验效果置信区间。draft 每 run 仅 1 个时不做组内 Vendi 对照。

`comparability.csv` 保留 GPU、初始 commit、训练 seed、预算、主 agent 模型等不一致或缺失提醒。不会按 seed 或时间临近创造配对，也不会借用另一轮 baseline。没有可验证 parent 的 baseline 不能产生对照 effect；单侧可用时最多生成描述性单 run 曲线。所有缺失需要连同曲线一起解释。

## 新任务、其他实验组与重复实验

读取 `run.json` 身份，并复用 `analyze_runs.py` 的命名回退和 `--manifest` CSV（`run,task_id,study_id,pair_id,arm,exclude_reason`）。新方案设置不同 study ID，独立重复使用不同 pair ID。重复有效/有源码 attempt 不择优配对。可选任意 arm 名称及参考组：

```bash
uv run --script compare_vendi.py --runs /path/to/runs --manifest comparisons.csv \
  --parent-map reviewed-parents.csv --arms control retrieval-v2 --baseline control \
  --out results/vendi-new-study

uv run --script compare_vendi.py --parent-map reviewed-parents.csv --valid-only \
  --out results/vendi-completed
```

## 缓存与离线重算

摘要/变更判断按 source、prompt、模型、API、endpoint 和上下文版本缓存；每次分片/合并或 diff 判断最多 3 次总尝试，证据修正与传输重试共享预算。缓存命中及已确认的 AST 等价不需要 API key。失败不缓存为有效结果，接受的证据不足判断会保留。

`samples.jsonl` 保存中性文本、embedding、来源引用和状态；不保存整份源文件。可以只重算统计，或替换**全部可评分向量**，不重新调用摘要模型：

```bash
uv run --script compare_vendi.py --input results/vendi/samples.jsonl \
  --no-plots --out results/vendi-offline

uv run --script compare_vendi.py --input results/vendi/samples.jsonl --reembed \
  --embedding-model <足够上下文的模型> --embedding-revision <固定版本> \
  --out results/vendi-reembedded
```

Embedding 超长显式记为错误，不截断。默认 MiniLM 的 512 窗口沿用 AKB 配置，是测量设置，不代表优于其默认训练窗口。切换模型必须重嵌入整组。已经确认的 no-change、证据不足及缺 parent 状态不会复活为评分样本。`--prepare-only` 导出没有提取过的文本，需重新从 `--runs` 提取，不能靠 `--input` 或 `--reembed` 补出源码证据。

也接受外部标准化 JSONL（必须有 `task,study_id,run_id,arm,candidate_id,stage`，可加 `pair_id`），`text` 必须已是统一口径的机制描述；它不会自动把任意代码或自然语言归一化。纯离线向量输入必须全部提供同一 `embedding_model` 身份。外部数据不自动获得源码证据认证。

## 输出与退出状态

输出包括 `parent_map.csv`、`coverage.csv`、`comparability.csv`、`samples.jsonl`、`change_evidence/*.json`、`run_scores.csv`、`comparisons.csv`、`manifest.json`、`REPORT.md`。有足够样本时生成 PNG/PDF 的 Vendi 曲线；有完整配对时增加 effect 曲线和最大共同 m 的 paired 图。每次更新仅清理本脚本上次登记、此次不再生成的图。

退出码 0：检查/准备成功，或评分完整；2：缺证据、提取/embedding 失败或没有可评分集合。全部确认 no-change 时输出 coverage 并成功退出，不制造零分。配置/格式错误也会非零退出。

离线回归：

```bash
python -m unittest discover -s tests -p '*vendi*.py' -v
```
