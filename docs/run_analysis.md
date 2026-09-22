# Autoresearch 结果分析与画图

仓库根目录的 `analyze_runs.py` 读取本机已下载的结果，不连接集群或运行训练。默认输入 `~/nautilus/autoresearch-result/`，默认输出本仓库 `results/analysis/`，不依赖当前 shell 的工作目录。

## 运行

```bash
cd /Users/william/Documents/project/python/autoresearch
uv run --script analyze_runs.py

# 自定义输入和输出
uv run --script analyze_runs.py --runs /path/to/downloaded-runs --out results/my-study

# 自定义参考组，或显式指定任意对照；--contrast 可重复
uv run --script analyze_runs.py --reference control
uv run --script analyze_runs.py --contrast draft:baseline --contrast draft-improve:draft
```

脚本声明了独立的 Matplotlib / SciPy 依赖，`uv run --script` 会创建脚本环境，不修改训练用的 `pyproject.toml`、`uv.lock` 或共享 venv。已有这两个包时也可直接 `python analyze_runs.py`。

## 图表与统计口径

- `charts/*_paired.png` / `.pdf`：每条线是一对独立实验，各 arm 使用该 run 内最优的有效 completed trial。纵轴保留原始验证分数；对于 loss 等越小越好的指标，纵轴反向。
- `charts/*_effect.png` / `.pdf`：每个点是一对实验的差值。最大化指标为 treatment − reference，最小化指标为 reference − treatment，始终正值表示 treatment 更好。两对及以上显示配对差值均值的 95% t 区间，一对不画区间，不做显著性判断。
- `run_inventory.csv` / `trial_inventory.csv`：所有 run / trial 的状态、排除原因、最佳指标和身份信息。没有有效结果的 run 会记录但不进入图表，不当作零分。
- `pairs.csv` / `effects.csv` / `pair_issues.csv`：实际配对、统计汇总及无法配对的原因。`summary.md` 提供文字说明，`analysis.json` 记录输入、参数、脚本 hash 和本次生成图表。

主数据取自 `trials/*/source.json` 和 `metrics.json`，要求 completed、身份匹配、源码/指标/配置及已绑定日志的 SHA-256 一致、存在验证预测、主指标 `score` 有限，且声明 `metric_version` 和布尔值 `maximize`。不从 TSV、日志或总结里猜分数。`best.json` 只作交叉检查，其四舍五入分数不覆盖原始指标。

这里分析的是**搜索后选出的最佳验证分数**，不重新评分、不读取测试集答案，也不证明预测文件与分数的数值关系。trial 数量不等于独立重复次数；区间中的 n 是实验对数。

## 新实验、任务和配对

优先读取 `run.json` 的 `task_id`、`study_id`、`pair_id`、`experiment_arm`（或 `arm`）。现有命名也可直接识别：

```text
run/20260921_084734-<task>-analogy                       -> task
jubias-pair-001-analogy-<Pod UUID>                      -> pair_id=jubias-pair-001
jubias-pair-001                                       -> study_id=jubias
```

同一 `(study_id, task_id, pair_id)` 下，每个被比较的 arm 必须恰好有一个有效 run。重启后产生多个有效 attempt 时不会自动取最新、取最高分或组合出多对；请明确划分配对。训练 seed 不能代替实验对编号。

无需改脚本，可通过 CSV 补充/覆盖身份或排除记录：

```csv
run,task_id,study_id,pair_id,arm,exclude_reason
downloaded-run-a,new-task,prompt-v2,repeat-001,baseline,
downloaded-run-b,new-task,prompt-v2,repeat-001,analogy,
downloaded-run-c,new-task,prompt-v2,repeat-002,baseline,
downloaded-run-d,new-task,prompt-v2,repeat-002,analogy,
failed-protocol-run,new-task,prompt-v2,repeat-003,analogy,wrong starting protocol
```

```bash
uv run --script analyze_runs.py --manifest comparisons.csv
```

`run` 必须是实际下载目录名；只需列出要覆盖的 run，空单元格保留自动识别值，未知目录名会报错。CSV 不允许修改分数、指标方向或数据身份。

配对内要求 prepared ID、任务内容 hash、评价器 hash、指标版本、优化方向和验证行数一致。不同 study、任务、任务内容、评价器、指标版本或方向分别出图，不混合平均。**同一 study 下，不同独立 pair 可以使用不同 prepared split/seed**，但每对内部必须匹配；如果它们属于不同方案，设置不同 `study_id`，即使任务名称相同也不会混在一起。

GPU、起始 commit、初始 seed、资源、环境 lock、预算或主 agent 模型不一致/缺失会列为可比性提醒，保留描述性图表。当前记录缺少结构化的 `budget_seconds` 和 `agent_model`；脚本不会从 six-hour Job 上限推断实际运行了六小时。提醒不是完整的实验合规审计。若要排除某轮，用 CSV 的 `exclude_reason` 明确记录理由。

后续更换任务需继续导出同一 trial/receipt 结构，并在指标文件中填写该任务的 `metric_version`、`score`、`maximize`；分析代码没有 Jigsaw 指标或 arm 列表的硬编码。每次重跑会更新输出表和图，并清理本脚本上次记录但本次不再适用的图；原始下载结果不变。无可配对数据时只输出审计表和说明，退出码为 1。

候选机制多样性另用根目录 `compare_vendi.py`，先运行 `python3 compare_vendi.py --prepare-only` 审查父节点映射。它保留失败训练的源码，与这里的验证分数分析使用不同的有效性标准；详见 [Vendi 分析说明](compare_vendi.md)。

要比较各 run 的完整解法集合，使用 `compare_solution_vendi.py`，不需要 parent 映射；见 [完整方案 Vendi 说明](solution_vendi.md)。
