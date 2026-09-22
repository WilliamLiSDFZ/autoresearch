# Autoresearch result analysis

Input: `/Users/william/nautilus/autoresearch-result`
Runs: 10; eligible: 6; excluded: 4.
Matched contrasts: 3; pairing issues: 0.

Scores are best verified completed **validation** results, not held-out test scores. Each independent experimental pair counts once per contrast; individual trials are not repeats. Effects are direction-corrected (positive means the comparison arm is better). 95% t intervals use variation across pairs and assume independent draws; n=1 has no interval. Different studies, tasks, metric versions, task hashes and evaluator hashes are never pooled. Different prepared splits may enter one cohort only through separate internally matched pairs.

- jubias / jigsaw-unintended-bias-in-toxicity-classification / analogy vs baseline: mean effect **-0.014826**, n=3, 95% CI [-0.061912, +0.032261].

## Comparability notes

- jubias-pair-001 (analogy vs baseline): agent_model_unknown;budget_seconds_unknown;gpu_differs;start_commit_differs.
- jubias-pair-002 (analogy vs baseline): agent_model_unknown;budget_seconds_unknown;gpu_differs.
- jubias-pair-003 (analogy vs baseline): agent_model_unknown;best_pointer_differs_from_verified_optimum;budget_seconds_unknown;gpu_differs.

GPU, initial commit, model and budget differences/unknowns are flags, not automatic exclusions. A flagged pair describes observed outcomes and does not isolate the treatment's causal effect. A completed trial does not prove that the entire run exhausted its intended budget. No timestamp proximity or shared training seed is used to invent pairings. Consult run_inventory.csv for identities/hardware, trial_inventory.csv for rejected candidates, and pair_issues.csv for ambiguous or incompatible pairs.
