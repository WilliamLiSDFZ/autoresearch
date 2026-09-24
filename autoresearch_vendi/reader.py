"""Read complete immutable solution snapshots without executing code."""

from collections import defaultdict
import hashlib
from pathlib import Path

from analyze_runs import digest, identify_run, load_overrides, read_json, read_trial


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_symlink_file:" + str(path))
    return path


def _json(path):
    return read_json(_regular(path))


def _clear_unsafe_pairs(samples, issues):
    groups = defaultdict(dict)
    for sample in samples:
        if sample["source"] and sample["pair_id"]:
            groups[(sample["study_id"], sample["task"], sample["pair_id"])][sample["run_id"]] = sample
    for (study, task, pair), runs in groups.items():
        arms = defaultdict(list)
        for run in runs.values():
            arms[run["arm"]].append(run["run_id"])
        reason = ""
        if any(len(names) > 1 for names in arms.values()):
            reason = "ambiguous_source_runs_per_arm"
        fields = ("prepared_id", "task_hash", "evaluator_sha256")
        mismatches = [field for field in fields if len({str(run.get(field, "")) for run in runs.values()}) > 1]
        for field in ("metric_version", "maximize"):
            known = {str(run[field]) for run in runs.values() if run.get(field) not in (None, "")}
            if len(known) > 1:
                mismatches.append(field)
            if any(run.get(field) in (None, "") for run in runs.values()):
                issues.append({"study_id": study, "task": task, "pair_id": pair,
                               "reason": "pair_" + field + "_unknown", "runs": ";".join(sorted(runs))})
        if mismatches and not reason:
            reason = "incompatible_pair:" + ",".join(mismatches)
        if reason:
            for sample in samples:
                if sample["run_id"] in runs:
                    sample["pair_id"] = ""
            issues.append({"study_id": study, "task": task, "pair_id": pair, "reason": reason,
                           "runs": ";".join(sorted(runs))})


# 只读加载完整方案，失败训练仍保留为静态分析候选。
def load_candidates(root: Path, manifest: Path | None = None):
    """Return verified complete solution candidates and run issues."""
    if not root.is_dir():
        raise ValueError("Result root must be an existing directory: " + str(root))
    overrides = load_overrides(manifest)
    paths = [root] if (root / "run.json").is_file() else sorted(p.parent for p in root.glob("*/run.json"))
    unknown = set(overrides) - {path.name for path in paths}
    if unknown:
        raise ValueError("Manifest references missing runs: " + ",".join(sorted(unknown)))
    samples, issues = [], []
    for run in paths:
        if run.is_symlink() or (run / "trials").is_symlink():
            issues.append({"run": run.name, "reason": "symlink_run_or_trials"})
            continue
        try:
            override = overrides.get(run.name, {})
            identity = identify_run(run, _json(run / "run.json"), override)
            if override.get("exclude_reason"):
                issues.append({"run": run.name, "reason": "manual:" + override["exclude_reason"]})
                continue
            if not all(identity.get(field) for field in ("task_id", "study_id", "pair_id", "arm")):
                identity["pair_id"] = ""
                issues.append({"run": run.name, "reason": "missing_pair_identity_use_manifest"})
            evaluator = run / "_worktree" / "prepare.py"
            evaluator_hash = digest(_regular(evaluator)) if evaluator.exists() else ""
        except (OSError, ValueError, TypeError) as exc:
            issues.append({"run": run.name, "reason": "invalid_run_metadata:" + str(exc)})
            continue
        candidates = []
        for trial in sorted((run / "trials").glob("*")):
            if not trial.is_dir():
                continue
            sample = {"task": identity["task_id"], "study_id": identity["study_id"], "run_id": run.name,
                      "arm": identity["arm"], "pair_id": identity["pair_id"], "candidate_id": trial.name,
                      "stage": "solution", "view": "implementation", "status": "unknown",
                      "is_valid": False, "source_hash": "", "source_refs": [str(trial / "source.py"), str(trial / "source.json")],
                      "source": "", "extraction_status": "missing_source",
                      "representation_version": "solution-v1", "evaluator_sha256": evaluator_hash,
                      "metric_version": "", "maximize": ""}
            for field in ("prepared_id", "task_hash", "gpu", "start_commit", "seed", "budget_seconds", "agent_model"):
                sample[field] = identity.get(field, "")
            try:
                if trial.is_symlink():
                    raise ValueError("symlink_trial")
                content = _regular(trial / "source.py").read_bytes()
                sample["source_hash"] = hashlib.sha256(content).hexdigest()
                receipt = _json(trial / "source.json")
                sample["status"] = receipt.get("execution_status") or "unknown"
                if receipt.get("protocol") != "autoresearch-experiment-v1":
                    raise ValueError("unknown_source_protocol")
                if receipt.get("run_id") != identity["run_id"] or receipt.get("experiment_id") != trial.name:
                    raise ValueError("source_identity_mismatch")
                if receipt.get("sha256") != sample["source_hash"]:
                    raise ValueError("source_hash_mismatch")
                if not identity["prepared_id"] or receipt.get("prepared_id") != identity["prepared_id"]:
                    raise ValueError("source_prepared_id_mismatch")
                sample["source"] = content.decode("utf-8")
                if sample["status"] == "completed":
                    try:
                        result = read_trial(trial, identity)
                        sample.update(is_valid=True, metric_version=result["metric_version"], maximize=result["maximize"])
                    except (OSError, ValueError, TypeError) as exc:
                        sample["validity_error"] = str(exc)
            except (OSError, ValueError, TypeError) as exc:
                sample["error"] = str(exc)
            sample["extraction_status"] = "pending" if sample["source"] else "missing_source"
            candidates.append(sample)
        if not any(sample["source"] for sample in candidates):
            issues.append({"run": run.name, "reason": "no_verified_source_candidates"})
        metric_ids = {(sample["metric_version"], sample["maximize"]) for sample in candidates if sample["is_valid"]}
        if len(metric_ids) > 1:
            issues.append({"run": run.name, "reason": "inconsistent_run_metric_identity"})
        for sample in candidates:
            if len(metric_ids) == 1:
                sample["metric_version"], sample["maximize"] = next(iter(metric_ids))
            elif len(metric_ids) > 1:
                sample["pair_id"] = ""
            samples.append(sample)
    _clear_unsafe_pairs(samples, issues)
    return samples, issues
