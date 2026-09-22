# Vendored unchanged from Agentic_Knowledge_Base/scripts/vendi_metrics.py.
# Source snapshot SHA256: 1930b2f363d735170ad45ec1beb8bafa704b7011ce6d1c9bf9d0b0fc77e443eb
"""Offline, equal-sample-size Vendi comparisons; subset bands are not effect CIs."""

from collections import defaultdict
import hashlib
import itertools
import math

import numpy as np


def _normalized_vectors(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("Embeddings must be a nonempty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings must contain only finite values")
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Zero embeddings have no cosine similarity")
    values = values / scale  # Avoid overflow/underflow in the Euclidean norm.
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _score_kernel(kernel):
    size = len(kernel)
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / (2 * size))
    if eigenvalues[0] < -1e-10:
        raise ValueError("Cosine kernel is not positive semidefinite")
    # Only repair eigenvalue roundoff; negative cosine entries are valid.
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues /= eigenvalues.sum()
    positive = eigenvalues[eigenvalues > 0]
    score = float(np.exp(-np.sum(positive * np.log(positive))))
    return min(float(size), max(1.0, score))


def vendi_score(vectors):
    """Return standard q=1 Vendi from a cosine kernel of nonzero row vectors.

    Repeated rows are observations and are deliberately retained. An individual
    vector has score 1. Empty, zero, nonfinite or ragged inputs are rejected.
    """
    values = _normalized_vectors(vectors)
    return _score_kernel(values @ values.T)


def _subset_summary(kernel, size, repeats, seed_key):
    count = len(kernel)
    if math.comb(count, size) <= repeats:
        subsets = itertools.combinations(range(count), size)
    else:
        digest = hashlib.sha256(seed_key.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        # Each draw is without replacement; different draws can repeat a subset.
        subsets = (rng.choice(count, size, replace=False) for _ in range(repeats))
    scores = []
    for subset in subsets:
        indices = np.asarray(subset, dtype=int)
        scores.append(_score_kernel(kernel[np.ix_(indices, indices)]))
    return {
        "vendi": float(np.mean(scores)),
        "subset_low": float(np.quantile(scores, 0.025)),
        "subset_high": float(np.quantile(scores, 0.975)),
        "n_subsets": len(scores),
    }


def compare_samples(samples, baseline="A", repeats=1000, seed=42):
    """Return (run_scores, comparisons) for already embedded candidate records.

    Required keys: task, run_id, arm, stage, view, embedding. pair_id is optional.
    Within each task/stage/view, runs with at least two candidates form one fixed
    cohort and share m=2..min(run sizes). All candidates, including identical
    embeddings, are retained. Input order does not affect seeded sampling.

    Unpaired arm means and explicitly paired differences are descriptive; no
    inferential confidence intervals or automatic baseline borrowing are used.
    subset_low/high describe candidate-subset variability only. Missing values
    are None. A missing/insufficient arm never contributes a zero score.
    """
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    groups = defaultdict(lambda: defaultdict(list))
    metadata = {}
    pair_runs = {}
    for sample in samples:
        for key in ("task", "run_id", "arm", "stage", "view", "embedding"):
            if key not in sample:
                raise ValueError(f"Sample is missing {key}")
        task, run_id, arm = (str(sample[key]) for key in ("task", "run_id", "arm"))
        pair_id = str(sample.get("pair_id") or "")
        run_key = (task, run_id)
        if run_key in metadata and metadata[run_key] != (arm, pair_id):
            raise ValueError(f"Inconsistent arm/pair_id for run {run_id}")
        metadata[run_key] = (arm, pair_id)
        if pair_id:
            pair_key = (task, pair_id, arm)
            if pair_key in pair_runs and pair_runs[pair_key] != run_id:
                raise ValueError(f"Multiple runs for task/pair_id/arm {pair_key}")
            pair_runs[pair_key] = run_id
        group_key = (task, str(sample["stage"]), str(sample["view"]))
        groups[group_key][run_id].append(sample["embedding"])

    run_scores, comparisons = [], []
    for group_key, runs in sorted(groups.items()):
        task, stage, view = group_key
        context = dict(task=task, stage=stage, view=view)
        kernels = {}
        dimensions = set()
        for run_id, vectors in sorted(runs.items()):
            values = _normalized_vectors(vectors)
            dimensions.add(values.shape[1])
            if len(values) >= 2:
                # Stable ordering preserves duplicates while removing file-order effects.
                values = np.asarray(sorted(values.tolist()), dtype=np.float64)
                kernels[run_id] = values @ values.T
        if len(dimensions) > 1:
            raise ValueError(f"Embedding dimensions differ within {group_key}")
        arms = sorted({metadata[(task, run_id)][0] for run_id in runs})
        sizes = range(2, min(map(len, kernels.values())) + 1) if kernels else [None]
        for size in sizes:
            by_arm = defaultdict(list)
            for run_id, kernel in kernels.items():
                arm, pair_id = metadata[(task, run_id)]
                summary = _subset_summary(kernel, size, repeats,
                                          repr((seed, group_key, run_id, size)))
                row = dict(context, run_id=run_id, arm=arm, pair_id=pair_id,
                           n_total=len(kernel), m=size, **summary)
                run_scores.append(row)
                by_arm[arm].append(row)
            baseline_rows = by_arm[baseline]
            for arm in arms:
                if arm == baseline:
                    continue
                rows = by_arm[arm]
                base = dict(context, m=size, arm=arm, baseline=baseline,
                            pair_id="", run_id="", baseline_run_id="",
                            n_runs=len(rows), n_baseline_runs=len(baseline_rows),
                            n_pairs=0, arm_vendi=None, baseline_vendi=None,
                            delta=None, status="descriptive")
                comparison = dict(base, kind="unpaired_mean")
                if rows:
                    comparison["arm_vendi"] = float(np.mean([r["vendi"] for r in rows]))
                if baseline_rows:
                    comparison["baseline_vendi"] = float(np.mean([r["vendi"] for r in baseline_rows]))
                if not baseline_rows:
                    comparison["status"] = ("missing_baseline" if baseline not in arms
                                            else "insufficient_candidates")
                elif not rows:
                    comparison["status"] = "insufficient_candidates"
                else:
                    comparison["delta"] = comparison["arm_vendi"] - comparison["baseline_vendi"]
                comparisons.append(comparison)

                paired = []
                arm_pairs = {r["pair_id"]: r for r in rows if r["pair_id"]}
                baseline_pairs = {r["pair_id"]: r for r in baseline_rows if r["pair_id"]}
                for pair_id in sorted(arm_pairs.keys() | baseline_pairs.keys()):
                    a, b = arm_pairs.get(pair_id), baseline_pairs.get(pair_id)
                    pair = dict(base, kind="pair", pair_id=pair_id,
                                run_id=a["run_id"] if a else "",
                                baseline_run_id=b["run_id"] if b else "",
                                n_runs=int(a is not None), n_baseline_runs=int(b is not None),
                                arm_vendi=a["vendi"] if a else None,
                                baseline_vendi=b["vendi"] if b else None,
                                n_pairs=int(a is not None and b is not None))
                    if a is not None and b is not None:
                        pair["delta"] = a["vendi"] - b["vendi"]
                        paired.append(pair)
                    else:
                        pair["status"] = "missing_pair"
                    comparisons.append(pair)
                if arm_pairs or baseline_pairs:
                    mean = dict(base, kind="paired_mean", n_pairs=len(paired),
                                n_runs=len(paired), n_baseline_runs=len(paired))
                    if paired:
                        for field in ("arm_vendi", "baseline_vendi", "delta"):
                            mean[field] = float(np.mean([r[field] for r in paired]))
                    else:
                        mean["status"] = "no_complete_pairs"
                    comparisons.append(mean)
    return run_scores, comparisons
