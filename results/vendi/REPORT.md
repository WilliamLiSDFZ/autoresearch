# Autoresearch Vendi analysis

Mode: conditional on evidenced static changes

Candidates: 179; available representations: 0; run issues: 11.

Each study/task/source-protocol/stage/view is separate. A fixed cohort of runs with at least two available candidates shares m=2..minimum candidate count. An unavailable arm never contributes zero. Runs with one available candidate have no matched-count comparison. Failed training candidates are retained unless valid-only is selected; static code is not proof of successful runtime activation.

Implementation diversity is conditional on evidenced changes. Read changed/no-change/insufficient/missing counts and fractions alongside Vendi. Identical mechanisms in different candidates retain their frequency. Missing parents are never replaced by the previous trial or nearest source. Evidence checks establish source provenance and static connections, not semantic correctness or scientific novelty.

Effects = comparison-arm Vendi minus reference Vendi. Positive means more diverse, not better. Subset bands are candidate-subset variability, NOT effect confidence intervals. Explicit experimental pair identity is required; training seeds and timestamp proximity do not create pairs. More independent, matched experiments are required for causal/significance claims. Budget is not automatically equalized.

## Coverage

| Run | Stage | Attempts | Available | Changed | No change | Insufficient | Missing / errors |
|---|---|---:|---:|---:|---:|---:|---:|
| jubias-pair-001-analogy-256b6bd9-6a19-4ffb-ba27-6df516483f0b | draft | 1 | 0 | 0 | 0 | 0 | 0 / 1 |
| jubias-pair-001-analogy-256b6bd9-6a19-4ffb-ba27-6df516483f0b | improve | 6 | 0 | 0 | 0 | 0 | 0 / 6 |
| jubias-pair-001-baseline-44e0ae1e-6af8-43bd-a941-c7d230dca3f7 |  | 10 | 0 | 0 | 0 | 0 | 10 / 0 |
| jubias-pair-001-baseline-44e0ae1e-6af8-43bd-a941-c7d230dca3f7 | draft | 1 | 0 | 0 | 0 | 0 | 0 / 1 |
| jubias-pair-002-analogy-750393d4-d146-487c-8f0f-f0ff710ff006 |  | 17 | 0 | 0 | 0 | 0 | 17 / 0 |
| jubias-pair-002-baseline-97784c12-1daf-4f23-930a-3c856920f958 |  | 8 | 0 | 0 | 0 | 0 | 8 / 0 |
| jubias-pair-002-baseline-97784c12-1daf-4f23-930a-3c856920f958 | draft | 1 | 0 | 0 | 0 | 0 | 0 / 1 |
| jubias-pair-003-analogy-289ef6cc-7122-4b20-a6df-6da2767d0636 |  | 10 | 0 | 0 | 0 | 0 | 10 / 0 |
| jubias-pair-003-baseline-529c8229-3944-4383-ad99-83c2b9d671d1 |  | 16 | 0 | 0 | 0 | 0 | 16 / 0 |
| jubias-pair-003-baseline-529c8229-3944-4383-ad99-83c2b9d671d1 | draft | 1 | 0 | 0 | 0 | 0 | 0 / 1 |
| jubias-pair-004-analogy-1d27d282-3a8c-483b-a230-06143ca33405 |  | 1 | 0 | 0 | 0 | 0 | 1 / 0 |
| jubias-pair-004-analogy-f9fd19d9-60e2-46ca-a494-870de58cddc4 |  | 13 | 0 | 0 | 0 | 0 | 13 / 0 |
| jubias-pair-004-baseline-0b96a385-1ed1-46ab-a40a-6748936a9a03 |  | 2 | 0 | 0 | 0 | 0 | 2 / 0 |
| jubias-pair-004-baseline-7cbc6d0e-1c6f-4092-a51e-69d2c716e82c |  | 16 | 0 | 0 | 0 | 0 | 16 / 0 |
| jubias-pair-005-analogy-221d48b3-c95c-4646-bdd6-924c1fd1d22f |  | 1 | 0 | 0 | 0 | 0 | 1 / 0 |
| jubias-pair-005-analogy-63fda2a5-3525-4c7c-b338-70771f48db71 |  | 16 | 0 | 0 | 0 | 0 | 16 / 0 |
| jubias-pair-005-baseline-cf3aa6fa-d637-47c0-82a7-f1993a3fd7b6 |  | 15 | 0 | 0 | 0 | 0 | 15 / 0 |
| jubias-pair-005-baseline-cf3aa6fa-d637-47c0-82a7-f1993a3fd7b6 | draft | 1 | 0 | 0 | 0 | 0 | 0 / 1 |
| jubias-pair-005-baseline-fd741ba9-6ba9-4d20-b607-bcd9a22c22e1 |  | 3 | 0 | 0 | 0 | 0 | 3 / 0 |
| jubias-pair-006-analogy-52bf396f-ae55-4c66-8193-32d654b36f85 |  | 15 | 0 | 0 | 0 | 0 | 15 / 0 |
| jubias-pair-006-analogy-89485097-7cf6-488d-b7e9-9e714a821a05 |  | 1 | 0 | 0 | 0 | 0 | 1 / 0 |
| jubias-pair-006-analogy-db60581a-f7ec-4c16-83c0-6fbb1379d6d8 |  | 2 | 0 | 0 | 0 | 0 | 2 / 0 |
| jubias-pair-006-baseline-2ef1b7bc-11e6-451b-ae43-cfc8fe26baa1 |  | 3 | 0 | 0 | 0 | 0 | 3 / 0 |
| jubias-pair-006-baseline-2ef1b7bc-11e6-451b-ae43-cfc8fe26baa1 | draft | 1 | 0 | 0 | 0 | 0 | 0 / 1 |
| jubias-pair-006-baseline-9c6629d3-6ce1-449a-974f-e7a80dae95ac |  | 18 | 0 | 0 | 0 | 0 | 18 / 0 |

## Comparability

- jubias-pair-001: gpu_differs;start_commit_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-002: gpu_differs;budget_seconds_unknown;agent_model_unknown.
- jubias-pair-003: gpu_differs;budget_seconds_unknown;agent_model_unknown.

See coverage.csv for source and extraction failures, parent_map.csv for ancestry inventory (including unresolved rows), and manifest.json for measurement settings. Generic JSONL imports are already prepared representations; they are not revalidated against original source. Use a separate output directory for sensitivity analyses.
