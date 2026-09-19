# autoresearch

This is an experiment to have the LLM do its own research on Jigsaw Unintended Bias in Toxicity Classification.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `codex/jigsaw-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b codex/jigsaw-<tag> codex/jigsaw-unintended-bias` from the Jigsaw task branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — upstream repository context; this branch uses the Jigsaw task.
   - `prepare.py` — fixed Jigsaw data preparation, train/validation split, and evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Run `uv run prepare.py verify` to check the prepared Jigsaw data in `results/jigsaw-data/` (or the directory selected by `AUTORESEARCH_JIGSAW_DIR`). If missing, run `uv run prepare.py prepare --data-dir /workspace/data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public`, adjusting the public data path if needed. Reuse the same prepared split and evaluator across comparison runs; preparation is not repeated for each experiment.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. There is currently **no fixed runtime limit** for an experiment or the overall research loop. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py` or the prepared data. They contain the fixed data split, labels, and evaluation. Fit learned preprocessing only on training rows; do not train on validation or private test labels.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_predictions` function in `prepare.py` is the ground truth metric: the Jigsaw composite AUC using continuous probabilities.

**The goal is simple: get the highest val_score.** Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The code must run successfully and use the fixed Jigsaw evaluation.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_score gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_score improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline. If `train.py` is still the upstream language-model script, first adapt it to Jigsaw classification using `load_data()` and `evaluate_predictions()` from `prepare.py`, and remove its upstream time-based stopping logic. `load_data()` returns training, validation, and test DataFrames; pass continuous probabilities in validation row order to `evaluate_predictions()` and print its returned `score` as `val_score`. Once a Jigsaw baseline exists, run it as is before making experimental changes.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_score:        0.900000
training_seconds: 842.3
total_seconds:    891.5
peak_vram_mb:     45060.2
```

Training duration depends on the model and training schedule. You can extract the key metric from the log file:

```
grep "^val_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.900000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_score	memory_gb	status	description
a1b2c3d	0.900000	44.0	keep	baseline
b2c3d4e	0.905000	44.2	keep	increase LR to 0.04
c3d4e5f	0.897000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `codex/jigsaw-mar5` or `codex/jigsaw-mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_score:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_score improved (higher), you "advance" the branch, keeping the git commit
9. If val_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The user then wakes up to experimental results, all completed by you while they slept!
