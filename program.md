# autoresearch: Jigsaw Unintended Bias in Toxicity Classification

Optimize a toxicity classifier on a fixed local validation set. **Maximize `score`**
from `prepare.py`; larger is better. Run one experiment at a time on the agreed
hardware. Treat validation as a development set, not an unbiased final test set.

## State of this task branch

`codex/jigsaw-unintended-bias` provides Jigsaw data preparation and evaluation.
**The upstream `train.py` still implements language-model pretraining and will not
run with this `prepare.py`. It must be rewritten for Jigsaw before experiments.**
The upstream README describes the original task, not this branch's workflow.

Prepare one shared Jigsaw baseline before forking comparable research runs:

1. Read this file, `prepare.py`, `train.py`, and the dependency configuration.
2. Prepare and verify one dataset contract using the commands below.
3. Adapt `train.py` once to load the fixed data, train a toxicity classifier,
   generate probabilities, save its checkpoint and predictions, and evaluate.
4. Run and validate that baseline. Save its metrics, data manifest hash, code
   commit, random seeds, dependency versions, hardware, and initialization assets.
5. Commit the shared baseline implementation. Record that immutable commit as
   the starting point for every comparison group. Do not independently develop
   a different baseline for each research agent or comparison group.

This common initialization is distinct from the experimental search. Decide
before comparisons whether its runtime and assets are excluded from the budget;
apply the same decision to every group. When comparing with MLEvolve, also align
its actual baseline, model/data access, split, scorer, resources, and budget.

## Fixed data preparation

Use public competition data already supplied by the user. Do not fetch or inspect
private MLE-bench labels, use an online hidden scoring service, or use test labels
for any stage of development. Keep data and outputs on Nautilus persistent
storage such as `/workspace`, not the container's ephemeral root filesystem.

Run preparation once before the comparison groups start:

```bash
uv run prepare.py prepare \
  --data-dir /workspace/data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public \
  --output-dir /workspace/autoresearch/results/jigsaw-data/seed-42 \
  --seed 42 \
  --validation-fraction 0.05

export AUTORESEARCH_JIGSAW_DIR=/workspace/autoresearch/results/jigsaw-data/seed-42
uv run prepare.py verify --prepared-dir "$AUTORESEARCH_JIGSAW_DIR"
```

The output contains `train.csv`, `validation.csv`, `test.csv`, `split.npz`, and
`manifest.json`. Reuse these exact artifacts across runs. The environment
variable selects the prepared directory; without it, the default is the
repository's `results/jigsaw-data` directory. Set it again in each new Pod or
configure it in the Pod definition.

**The same seed does not guarantee the same split across implementations.**
The standalone preparer uses its own deterministic NumPy stratified split.
For an existing MLEvolve comparison, add
`--mlevolve-contract /absolute/path/to/existing/contract` to the preparation
command to import its supported split contract. Verify source data, resulting
row IDs, and scorer compatibility before treating the groups as comparable.
Omit `--seed` and `--validation-fraction` to inherit the imported contract's
settings; explicitly supplied values must match. Use a separate output directory
for each different contract or seed.
Do not silently generate a new split when the intended contract cannot be used.

Preparation is independent of a particular model/tokenizer. Changing a model
may require new model-dependent caches, but must preserve these fixed row IDs
and labels. Fit vocabularies, feature statistics, and other learned transforms
only on training rows. Key derived caches by the data contract and transform
configuration; never reuse an incompatible cache.

## Training and evaluation interface

`train.py` should use these fixed helpers:

```python
from prepare import load_data, evaluate_predictions

train_df, validation_df, test_df = load_data()
# Fit only on train_df; produce probabilities in validation_df row order.
metrics = evaluate_predictions(validation_probabilities)
print(f"score: {metrics['score']:.8f}")
```

Both helpers accept `prepared_dir` explicitly when needed. Training and
validation tables contain `id`, `comment_text`, `target`, and the fixed identity
columns. The test table contains `id` and `comment_text`. Training may use soft
toxicity targets; scoring uses the scorer's fixed threshold at 0.5.

For durable, independently checkable scoring, write
`validation_predictions.csv` with exactly `id,prediction`, in validation row
order, with one finite probability in `[0, 1]` per row. Then run:

```bash
uv run prepare.py evaluate \
  --prepared-dir "$AUTORESEARCH_JIGSAW_DIR" \
  --predictions results/RUN_TAG/EXPERIMENT_ID/validation_predictions.csv \
  --output results/RUN_TAG/EXPERIMENT_ID/metrics.json
```

The fixed scorer version is `jubias-continuous-auc-v1`. It uses continuous
prediction scores, with ties handled by AUC ranking, rather than thresholding
predictions into classes. The final score is 25% overall AUC plus 75% bias score.
The bias score averages the exponent -5 power means of subgroup AUC, BPSN AUC
(background positive, subgroup negative), and BNSP AUC (background negative,
subgroup positive) across these nine identity groups:

`male`, `female`, `homosexual_gay_or_lesbian`, `christian`, `jewish`, `muslim`,
`black`, `white`, and `psychiatric_or_mental_illness`.

Identity membership uses the fixed threshold at 0.5. Use the implementation's
exact missing-value and validity rules. An invalid or undefined metric is a
failed evaluation; do not drop groups or substitute fabricated values. Compare
only scores from the same verified data contract and scorer version.

## Research run setup and boundaries

After the shared baseline is committed, create a fresh run branch from it:

```bash
git switch -c codex/jigsaw-RUN_TAG codex/jigsaw-unintended-bias
```

Use the recorded baseline commit instead of the branch name if that branch has
moved. Preserve unrelated user changes. Keep run artifacts under
`results/RUN_TAG/` and initialize `results/RUN_TAG/results.tsv`.

Before starting, establish the authorized stopping condition: total runtime,
experiment count, or an explicitly authorized unbounded run. Do not assume
unlimited execution or invent a numerical budget. Proceed under an already
specified budget without requesting another confirmation.

During experiments:

- Modify `train.py` for model architecture, optimization, features, and training.
- Keep `prepare.py`, the prepared data, split, scoring rules, and this protocol
  fixed. Do not train on validation/test labels or incorporate validation rows
  into preprocessing fits. Do not tune against hidden test results.
- Use the installed, agreed dependency set. Do not install new packages or
  change dependency configuration during search. Any needed additions belong
  in common initialization before freezing all comparison groups.
- Use only agreed pretrained assets and external data; preload and share them
  under the same rules across groups. Log seeds and relevant configuration.
- Maintain the agreed GPU count, hardware allocation, and parallelism.

## Time-controlled comparisons

This Markdown file and `prepare.py` **do not enforce a runtime deadline**.
Before a time-controlled comparison, configure an external supervisor/watchdog
to control the agent and all training subprocesses. No supervisor is supplied
by this task adaptation. Do not claim a strict limit without one.

For an end-to-end budget, use one monotonic deadline that includes agent
reasoning, edits, tool/model waits, initialization, compilation, training,
evaluation, artifact writes, and retries. Failures never reset that deadline.
Set any per-experiment limits separately, consistently across groups. Reserve
time for evaluation and durable output; do not start work that cannot finish.

Only accept results whose full evaluation and required artifacts completed
before the deadline. At expiry, stop new work, have the supervisor cancel
remaining processes, and return the best previously valid result. Log timeout
and cleanup latency separately. If no valid result finished, report failure.
If the protocol instead permits evaluation after search, call it a search-only
budget and fix that rule for every group in advance.

## Experiment loop and records

1. Check the branch, verified data contract, remaining budget, and current best.
2. Form one hypothesis, change `train.py`, and record its exact code commit.
3. Run `uv run train.py` under the configured supervisor, redirecting output to
   that experiment's own `run.log`. Save configuration, checkpoint, validation
   predictions, metrics, elapsed time, and peak memory in the same directory.
4. Verify successful completion and score the saved predictions with the fixed
   evaluator. Append the result to the durable TSV immediately.
5. Keep a higher valid score. Prefer simplicity on a true tie under the agreed
   tie rule. Otherwise restore the prior best `train.py` without removing logs,
   candidate commits, or unrelated changes. Record failed attempts and retries.
6. Continue until the authorized stopping condition, then report the best
   commit, score, contract, artifact paths, and consumed budget.

Use this tab-separated header; leave unavailable numeric fields empty, never
encode a crash as a zero score:

```text
experiment_id\tcommit\tscore\telapsed_seconds\tpeak_vram_mb\tstatus\tartifact_dir\tdescription
```

Statuses are `baseline`, `keep`, `discard`, `crash`, or `timeout`. Keep these
records and per-experiment artifacts outside destructive Git rollback paths;
`results/` is the intended persistent, untracked output directory. Save the
current best after each completed experiment instead of waiting until the end.
