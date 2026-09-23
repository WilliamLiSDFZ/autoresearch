# Autoresearch solution Vendi analysis

Mode: whole-candidate static solution summaries; no parent required

Candidates: 165; available representations: 165; run issues: 11.

Each study/task/source-protocol/stage/view is separate. A fixed cohort of runs with at least two available candidates shares m=2..minimum candidate count. An unavailable arm never contributes zero. Runs with one available candidate have no matched-count comparison. Failed training candidates are retained unless valid-only is selected; static code is not proof of successful runtime activation.

Complete source snapshots are summarized independently using one neutral prompt. Initial and later candidates share stage=solution; parent links, retrieval reports and performance scores are not summary inputs. Repeated solutions retain their frequency. This measures diversity of attempted implementations, not all proposed ideas, literature novelty or successful execution. Full-run scores use unequal counts and are descriptive; use matched-count scores for comparisons.

Effects = comparison-arm Vendi minus reference Vendi. Positive means more diverse, not better. Subset bands are candidate-subset variability, NOT effect confidence intervals. Explicit experimental pair identity is required; training seeds and timestamp proximity do not create pairs. More independent, matched experiments are required for causal/significance claims. Budget is not automatically equalized.

## Coverage

| Run | Attempts | Available | Verified completed | Pending extraction | Missing / errors |
|---|---:|---:|---:|---:|---:|
| jubias-pair-001-analogy-256b6bd9-6a19-4ffb-ba27-6df516483f0b | 7 | 7 | 7 | 0 | 0 / 0 |
| jubias-pair-001-baseline-44e0ae1e-6af8-43bd-a941-c7d230dca3f7 | 11 | 11 | 6 | 0 | 0 / 0 |
| jubias-pair-002-analogy-750393d4-d146-487c-8f0f-f0ff710ff006 | 17 | 17 | 16 | 0 | 0 / 0 |
| jubias-pair-002-baseline-97784c12-1daf-4f23-930a-3c856920f958 | 9 | 9 | 8 | 0 | 0 / 0 |
| jubias-pair-003-analogy-289ef6cc-7122-4b20-a6df-6da2767d0636 | 10 | 10 | 7 | 0 | 0 / 0 |
| jubias-pair-003-baseline-529c8229-3944-4383-ad99-83c2b9d671d1 | 17 | 17 | 16 | 0 | 0 / 0 |
| jubias-pair-004-analogy-f9fd19d9-60e2-46ca-a494-870de58cddc4 | 13 | 13 | 13 | 0 | 0 / 0 |
| jubias-pair-004-baseline-7cbc6d0e-1c6f-4092-a51e-69d2c716e82c | 16 | 16 | 16 | 0 | 0 / 0 |
| jubias-pair-005-analogy-63fda2a5-3525-4c7c-b338-70771f48db71 | 16 | 16 | 15 | 0 | 0 / 0 |
| jubias-pair-005-baseline-cf3aa6fa-d637-47c0-82a7-f1993a3fd7b6 | 16 | 16 | 15 | 0 | 0 / 0 |
| jubias-pair-006-analogy-52bf396f-ae55-4c66-8193-32d654b36f85 | 15 | 15 | 13 | 0 | 0 / 0 |
| jubias-pair-006-baseline-9c6629d3-6ce1-449a-974f-e7a80dae95ac | 18 | 18 | 17 | 0 | 0 / 0 |

## Comparability

- jubias-pair-001: gpu_differs;start_commit_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-002: gpu_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-003: gpu_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-004: gpu_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-005: gpu_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-006: gpu_differs;budget_seconds_unknown;agent_model_unknown.

See coverage.csv for source and extraction failures, solution_cards.csv for source-grounded summaries, and manifest.json for measurement settings. Generic JSONL imports are already prepared representations; they are not revalidated against original source. Use a separate output directory for sensitivity analyses.
