"""Read immutable candidate snapshots and explicit ancestry without executing code."""

from collections import defaultdict
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import re

from analyze_runs import digest, identify_run, load_overrides, read_json, read_trial


MAP_FIELDS = ("run", "candidate_id", "stage", "parent_id", "child_sha256", "parent_sha256", "evidence")


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_symlink_file:" + str(path))
    return path


def _json(path):
    return read_json(_regular(path))


def _parent_map(path, known):
    if path is None:
        return {}
    with _regular(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not set(MAP_FIELDS) <= fields or fields - set(MAP_FIELDS) - {"status", "reason"}:
            raise ValueError("Parent map columns: " + ",".join(MAP_FIELDS))
        result = {}
        for raw in reader:
            location = f"Parent map {path}:{reader.line_num}"
            if None in raw:
                raise ValueError(location + ": invalid CSV row")
            row = {key: (raw.get(key) or "").strip() for key in MAP_FIELDS}
            key = row["run"], row["candidate_id"]
            location += " (" + "/".join(key) + ")"
            if key not in known or key in result:
                raise ValueError(location + ": unknown or duplicate candidate")
            invalid = [field for field in ("child_sha256", "evidence") if not row[field]]
            if row["stage"] not in {"draft", "improve"}:
                invalid.insert(0, "stage")
            if invalid:
                raise ValueError(location + ": missing or invalid " + ", ".join(invalid) + ". "
                                 "Generated parent_map.csv includes unresolved inventory rows. "
                                 "Omit --parent-map to use recorded evidence automatically, or supply a separate "
                                 "reviewed copy containing only evidence-backed rows. Unresolved candidates remain missing.")
            if row["stage"] == "draft":
                if row["parent_id"] or row["parent_sha256"]:
                    raise ValueError(location + ": draft rows must not declare a parent")
            elif not row["parent_id"] or not row["parent_sha256"]:
                raise ValueError(location + ": improve rows require parent_id and parent_sha256")
            result[key] = row
        return result


def _initial_trial(run, candidate, source_hash):
    path = run / "results.tsv"
    if not path.exists():
        return False
    with _regular(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            description = row.get("description") or ""
            explicit_id = re.search(r"(?<![\w-])" + re.escape(candidate) + r"(?![\w-])", description)
            commit = row.get("commit") or ""
            matching_hash = len(commit) >= 7 and source_hash.startswith(commit)
            if (explicit_id or matching_hash) and re.search(r"\b(initial|draft)\b", description, re.I):
                return True
    return False


def _declared_parent(receipt):
    values = [receipt[key] for key in ("parent_id", "parent_experiment_id", "parent") if receipt.get(key)]
    if any(not isinstance(value, str) for value in values) or len(set(values)) > 1:
        raise ValueError("conflicting_source_parent_fields")
    return values[0] if values else ""


def _ancestry(run, candidate, receipt, source_hash, first, nodes):
    parent = _declared_parent(receipt)
    stage = receipt.get("stage") or ("improve" if parent else "")
    if stage not in {"", "draft", "improve"} or (stage == "draft" and parent):
        raise ValueError("invalid_source_stage")
    evidence = ["source.json"] if stage or parent else []
    expected_parent_hash = receipt.get("parent_sha256") or receipt.get("parent_source_sha256") or ""
    references = []
    adoption_path = run / "trials" / candidate / "adoption.json"
    if adoption_path.exists():
        adoption = _json(adoption_path)
        call_id = adoption.get("call_id") or ""
        if not isinstance(call_id, str) or not re.fullmatch(r"(draft|improve)-[\w-]+", call_id):
            raise ValueError("invalid_adoption_call_id")
        call_stage = call_id.split("-", 1)[0]
        adoption_parent = adoption.get("parent") or ""
        if not isinstance(adoption_parent, str):
            raise ValueError("invalid_adoption_parent")
        if (stage and stage != call_stage) or (parent and parent != adoption_parent):
            raise ValueError("source_adoption_ancestry_conflict")
        stage, parent = call_stage, adoption_parent
        references.append(str(adoption_path))
        evidence.append("adoption.json:" + call_id)
        if stage == "draft" and parent:
            raise ValueError("draft_has_parent")
        if stage == "improve":
            context_path = run / "analogy" / call_id / "context.json"
            manifest_path = run / "analogy" / call_id / "manifest.json"
            context, manifest = _json(context_path), _json(manifest_path)
            references.extend([str(context_path), str(manifest_path)])
            context_meta, manifest_meta = context.get("metadata", {}), manifest.get("input_metadata", {})
            if not isinstance(context_meta, dict) or not isinstance(manifest_meta, dict):
                raise ValueError("invalid_analogy_parent_metadata")
            for payload, metadata in ((context, context_meta), (manifest, manifest_meta)):
                if payload.get("stage") != "improve" or metadata.get("parent_experiment_id") != parent:
                    raise ValueError("analogy_parent_evidence_conflict")
                if metadata.get("run_id") != receipt.get("run_id"):
                    raise ValueError("analogy_parent_run_mismatch")
                if parent not in nodes or metadata.get("source_sha256") != nodes[parent]["source_hash"]:
                    raise ValueError("analogy_parent_hash_mismatch")
            if expected_parent_hash and expected_parent_hash != context_meta["source_sha256"]:
                raise ValueError("source_analogy_parent_hash_conflict")
            expected_parent_hash = context_meta["source_sha256"]
            evidence.extend(["context.metadata", "manifest.input_metadata"])
    if not stage and candidate == first and _initial_trial(run, candidate, source_hash):
        stage = "draft"
        evidence.append("results.tsv:initial first trial")
        references.append(str(run / "results.tsv"))
    return stage, parent, expected_parent_hash, ";".join(evidence), references


def _parent_error(candidate, parent, nodes, expected_hash):
    if parent == candidate:
        return "self_parent"
    if parent not in nodes or "/" in parent or "\\" in parent:
        return "unknown_or_cross_run_parent"
    node, previous = nodes[candidate], nodes[parent]
    if not previous["source"]:
        return "parent_source_unverified"
    if expected_hash and expected_hash != previous["source_hash"]:
        return "parent_hash_mismatch"
    child_time = node["receipt"].get("created_at_utc")
    parent_time = previous["receipt"].get("created_at_utc")
    if child_time and parent_time:
        try:
            if datetime.fromisoformat(parent_time) > datetime.fromisoformat(child_time):
                return "parent_created_after_child"
        except (TypeError, ValueError):
            return "invalid_parent_creation_timestamps"
    return ""


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


# 只读加载完整方案或可验证的父子变更，失败训练仍保留为静态分析候选。
def load_candidates(root: Path, manifest: Path | None = None, parent_map: Path | None = None, *, mode="changes"):
    """Return verified source candidates, run issues and optional ancestry inventory."""
    if mode not in {"changes", "solutions"}:
        raise ValueError("Candidate mode must be changes or solutions")
    if mode == "solutions" and parent_map is not None:
        raise ValueError("Solution analysis does not accept a parent map")
    if not root.is_dir():
        raise ValueError("Result root must be an existing directory: " + str(root))
    overrides = load_overrides(manifest)
    paths = [root] if (root / "run.json").is_file() else sorted(p.parent for p in root.glob("*/run.json"))
    unknown = set(overrides) - {path.name for path in paths}
    if unknown:
        raise ValueError("Manifest references missing runs: " + ",".join(sorted(unknown)))
    mappings = {}
    if mode == "changes":
        known = {(path.name, trial.name) for path in paths for trial in (path / "trials").glob("*") if trial.is_dir()}
        mappings = _parent_map(parent_map, known)
    samples, issues, parent_rows = [], [], []
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
        nodes = {}
        for trial in sorted((run / "trials").glob("*")):
            if not trial.is_dir():
                continue
            sample = {"task": identity["task_id"], "study_id": identity["study_id"], "run_id": run.name,
                      "arm": identity["arm"], "pair_id": identity["pair_id"], "candidate_id": trial.name,
                      "parent_id": "", "stage": "", "view": "implementation", "status": "unknown",
                      "is_valid": False, "source_hash": "", "source_refs": [str(trial / "source.py"), str(trial / "source.json")],
                      "source": "", "parent_source": "", "extraction_status": "missing_source",
                      "representation_version": "diff-v1", "evaluator_sha256": evaluator_hash,
                      "metric_version": "", "maximize": ""}
            for field in ("prepared_id", "task_hash", "gpu", "start_commit", "seed", "budget_seconds", "agent_model"):
                sample[field] = identity.get(field, "")
            receipt = {}
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
            nodes[trial.name] = {**sample, "receipt": receipt}
        if not any(node["source"] for node in nodes.values()):
            issues.append({"run": run.name, "reason": "no_verified_source_candidates"})
        metric_ids = {(node["metric_version"], node["maximize"]) for node in nodes.values() if node["is_valid"]}
        if len(metric_ids) > 1:
            issues.append({"run": run.name, "reason": "inconsistent_run_metric_identity"})
        for candidate, node in nodes.items():
            if len(metric_ids) == 1:
                node["metric_version"], node["maximize"] = next(iter(metric_ids))
            elif len(metric_ids) > 1:
                node["pair_id"] = ""
            if mode == "solutions":
                node.update(stage="solution", representation_version="solution-v1",
                            extraction_status="pending" if node["source"] else "missing_source")
                node.pop("receipt")
                samples.append(node)
                continue
            mapping = mappings.get((run.name, candidate))
            stage, parent, parent_hash, evidence = "", "", "", ""
            try:
                if node["source"]:
                    stage, parent, parent_hash, evidence, references = _ancestry(
                        run, candidate, node["receipt"], node["source_hash"], next(iter(nodes)), nodes)
                    node["source_refs"].extend(references)
                if mapping:
                    if mapping["child_sha256"] != node["source_hash"]:
                        raise ValueError("parent_map_child_hash_mismatch")
                    if (stage and mapping["stage"] != stage) or (parent and mapping["parent_id"] != parent):
                        raise ValueError("parent_map_conflicts_with_recorded_ancestry")
                    if parent_hash and mapping["parent_sha256"] != parent_hash:
                        raise ValueError("parent_map_parent_hash_conflict")
                    stage, parent, parent_hash, evidence = (mapping[key] for key in ("stage", "parent_id", "parent_sha256", "evidence"))
                if stage == "improve":
                    problem = _parent_error(candidate, parent, nodes, parent_hash)
                    if problem:
                        raise ValueError(problem)
                elif stage != "draft":
                    raise ValueError("missing_stage_or_parent_evidence")
                node.update(stage=stage, parent_id=parent, representation_version="summary-v1" if stage == "draft" else "diff-v1")
                if node["source"]:
                    node["extraction_status"] = "pending"
                    if parent:
                        node["parent_source"] = nodes[parent]["source"]
                        node["source_refs"].append(str(run / "trials" / parent / "source.py"))
            except (OSError, ValueError, TypeError) as exc:
                if mapping:
                    raise ValueError(run.name + "/" + candidate + ": " + str(exc)) from exc
                node["error"] = node.get("error") or str(exc)
            node["parent_evidence"] = evidence
        if mode == "solutions":
            continue
        for candidate, node in nodes.items():
            visited, current = set(), candidate
            while current in nodes and nodes[current]["parent_id"]:
                if current in visited:
                    if any((run.name, name) in mappings for name in visited):
                        raise ValueError(run.name + ": parent_cycle")
                    node.update(extraction_status="missing_source", error="parent_cycle", parent_source="")
                    break
                visited.add(current)
                current = nodes[current]["parent_id"]
            parent = node["parent_id"]
            parent_rows.append({"run": run.name, "candidate_id": candidate, "stage": node["stage"],
                                "parent_id": parent, "child_sha256": node["source_hash"],
                                "parent_sha256": nodes[parent]["source_hash"] if parent in nodes else "",
                                "evidence": node["parent_evidence"], "status": "verified" if node["extraction_status"] == "pending" else "missing",
                                "reason": node.get("error", "")})
            node.pop("receipt")
            samples.append(node)
    _clear_unsafe_pairs(samples, issues)
    return samples, issues, parent_rows
