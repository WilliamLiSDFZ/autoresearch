# 完整方案 Vendi 分析

`compare_solution_vendi.py` 衡量每个 run 尝试过的完整解法是否多样。不需要 `parent_map.csv`，也不读取 analogy 的检索报告或采纳声明。它与原来的 `compare_vendi.py` 变更分析共享源码验证、统计和绘图代码，默认输出与缓存分开保存。

## 运行

在仓库根目录执行：

```bash
# 只清点，不写文件、不调用模型
python3 compare_solution_vendi.py --dry-run

# 保存清点表和报告；不调用 API 或下载模型
python3 compare_solution_vendi.py --prepare-only

# 配置下面的 API 环境变量后，提取摘要、计算分数并画图
uv run --script compare_solution_vendi.py
```

默认读取 `~/nautilus/autoresearch-result`，输出到 `results/vendi-solutions/`，缓存到 `results/vendi-solutions-cache/`。`uv run --script` 创建独立分析环境，不修改训练用的 `uv.lock`。等价入口是 `compare_vendi.py --mode solutions`。

不要传 `--parent-map` 或 `--stages`。每个 trial 都是一个完整方案，首次实现与后续实现共同进入 `stage=solution` 的集合。旧 diff 输出或摘要不能作为完整方案输入，脚本会拒绝混用两种分析目录。

## 模型和接口

| 配置 | 默认值及来源 |
|---|---|
| 摘要模型 | `--summary-model gpt-5.6-terra` |
| 摘要 API | `--summary-api responses`，也支持 `chat` |
| 摘要密钥 | `VENDI_API_KEY` → `ANALOGY_API_KEY` → `OPENAI_API_KEY` → `LLM_API_KEY` |
| 摘要地址 | `--base-url` → `VENDI_BASE_URL` → `ANALOGY_BASE_URL` → `OPENAI_BASE_URL` → `LLM_BASE_URL` → OpenAI 默认地址 |
| Embedding | `--embedding-backend openai --embedding-model text-embedding-3-small` |
| Embedding 密钥 | `VENDI_EMBEDDING_API_KEY`；未设置时使用摘要密钥 |
| Embedding 地址 | `--embedding-base-url` → `VENDI_EMBEDDING_BASE_URL`；未设置时使用摘要地址 |

如果现有代理只支持生成文本，需单独配置支持 `/embeddings` 的地址和密钥。脚本不会从其他项目或实验日志搜集凭据，不会自动更换模型。运行时会把候选源码发送到摘要接口，把中性摘要发送到 embedding 接口。

也可选本地 MiniLM embedding；首次运行可能下载模型，需要增加独立脚本依赖：

```bash
uv run --with 'sentence-transformers>=5.0' --script compare_solution_vendi.py \
  --embedding-backend local --out results/vendi-solutions-local
```

本地后端仍需摘要 API。不同 embedding 模型的分数分开保存，不能直接与论文或另一后端的绝对值比较。已有缓存可复用；命中全部摘要与向量缓存时无需 API key。

若摘要已经生成、embedding 请求却返回 HTTP 404，通常需要为向量配置单独的接口和模型。脚本会显示安全的异常类型、状态码及尝试次数，不打印密钥或服务端正文。也可复用保存的摘要，显式改用本地 MiniLM，不再调用摘要 API：

```bash
uv run --with 'sentence-transformers>=5.0' --script compare_solution_vendi.py \
  --input results/vendi-solutions/samples.jsonl \
  --embedding-backend local --out results/vendi-solutions-local
```

该命令无需 `--env-file`。如果输入已有部分向量，另加 `--reembed`；全部候选都在嵌入前失败时不需要。MiniLM 首次使用需要下载，模型已缓存时可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 禁止模型联网。

## 提取和计算口径

1. 验证每个 `trials/*/source.py` 与 `source.json` 的源码 hash、run/trial 身份和 prepared ID。空 run 仅列为缺失记录。
2. 对每份完整源码提取统一的英文方法标题、最多 160 词的机制摘要和 `SOURCE:行号` 证据。长源码按片提取后合并，不静默丢弃尾部。摘要不接收实验组、指标成绩、父节点或检索报告；源码内的评论/名称也不得被当成指令或创新证据。
3. 对 `summary` 字段嵌入。标题与证据单独保存供审阅，不加入主分析向量。
4. 计算 cosine 相似度矩阵的标准 q=1 Vendi。重复机制在不同 trial 中保留频次，同一候选重复导出才去重。
5. 每个 study/task/源码协议组内，至少有两个可用候选的 run 构成固定集合，共用 `m=2..最少候选数`。组合数不超过 `--repeats`（默认 1000）时穷举，否则进行固定 seed 的不放回子集抽样。

例如 baseline 有 11 个有效摘要、analogy 有 7 个时，m=7 对 baseline 的 330 个子集全部计算，analogy 使用唯一的 7 个候选集合。取子集 Vendi 均值进行配对比较；图中子集范围不是实验效果置信区间。全量分数另存 `full_run_scores.csv`，候选数不同时仅作描述。

主分析保留有可信源码的失败/pending 训练候选，不将它们视为成功。只看具有完整成功回执的候选可以另跑敏感性分析：

```bash
uv run --script compare_solution_vendi.py --valid-only \
  --out results/vendi-solutions-completed
```

缺源码、摘要失败和 embedding 失败计入 coverage，不能补零。某些候选无法提取时，两组共享 m 会按实际可评分数量确定，需同时查看覆盖率。单候选全量 Vendi 为 1，但不足以形成匹配数量对照。

## 多任务与重复实验

沿用 `analyze_runs.py` 的身份解析及 `--manifest` CSV：`run,task_id,study_id,pair_id,arm,exclude_reason`。同一 pair/arm 出现多个有源码的 attempt 时不自动取最新或最高分。不同任务和 study 分开计算；训练 seed 不是 pair ID。

```bash
uv run --script compare_solution_vendi.py --runs /path/to/runs \
  --manifest comparisons.csv --arms control retrieval-v2 --baseline control \
  --out results/vendi-solutions-study2
```

GPU、起始 commit、预算和主 agent 模型等差异列在 `comparability.csv`。独立 run 才能提供实验重复；同一 run 的 trial 或抽样子集不能当作独立实验。当前输出是描述性比较，不提供因果结论或显著性检验。

## 输出与复算

- `solution_cards.csv`：每个候选的方法标题、机制摘要、证据、源码 hash 和状态。
- `samples.jsonl`：统一摘要、来源和向量；不保存整份源码。
- `coverage.csv`、`comparability.csv`：提取覆盖率及实验条件差异。
- `full_run_scores.csv`：每个 run 的全量描述性 Vendi。
- `run_scores.csv`、`comparisons.csv`：相同候选数下的分数及 treatment − baseline 差值。
- `*_vendi.*`、`*_effect.*`、`*_paired.*`：PNG/PDF 图；有足够样本与配对时生成。
- `manifest.json`、`REPORT.md`：运行参数、模型身份、代码 hash、摘要 API 调用计数与解释。

保存了完整向量后可离线重算，不调用摘要或 embedding：

```bash
uv run --script compare_solution_vendi.py \
  --input results/vendi-solutions/samples.jsonl --out results/vendi-solutions-recomputed
```

若 embedding 接口失败而摘要已生成，修好配置后可使用该 `--input` 重试嵌入。若部分候选已有向量、部分失败，必须加 `--reembed` 统一处理，不能混用现有向量与待嵌入文本；已验证的 embedding 缓存仍可复用。缓存文件损坏时需修复该缓存或指定新的 `--cache` 目录。`--prepare-only` 没有摘要，必须从 `--runs` 正式提取，不能从空白摘要反推源码。

检查/准备成功退出 0；正式运行缺证据、接口失败或没有足够候选退出 2，已完成的摘要和审计表仍保存。脚本不连接集群、不执行候选代码。

## 与论文的关系

依据 [Unlocking LLM Creativity in Science through Analogical Reasoning](https://arxiv.org/abs/2605.11258) 的 cosine Vendi、重复候选保留和问题内计算方式。作者公开实现嵌入解法标题；这里从完整代码提取机制摘要，所以是针对代码实验的适配，并非论文数值复现。

测量结论应写为“实际尝试的实现是否更加多样”，不能由 Vendi 单独判断科学原创性、所有尚未实现的想法，或解法是否更有效。性能分析与文献新颖性评估是另外两项任务。
