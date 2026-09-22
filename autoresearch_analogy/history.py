"""Read-only, same-run experiment history captured for one analogy episode."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "analogy-history-v1"
_ID = re.compile(r"[A-Za-z0-9_-]+")


def _time(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("History timestamps must be timezone-aware ISO timestamps")
    if result.tzinfo is None:
        raise ValueError("History timestamps require a timezone")
    return result.astimezone(timezone.utc)


def _inside(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("History path escapes the specified run directory")
    return path


def _read(root, path, sources):
    path = _inside(root, path)
    content = path.read_bytes()
    sources[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
    return content.decode("utf-8")


def _object(root, path, sources):
    value = json.loads(_read(root, path, sources))
    if not isinstance(value, dict):
        raise ValueError(f"History requires an object: {path.name}")
    return value


def _reference(root, value, run_id, category):
    """Resolve only explicit local or relocated same-run artifact references."""
    if not isinstance(value, str) or not value:
        raise ValueError("History artifact reference must be a nonempty string")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("History artifact references cannot contain traversal")
    if path.is_absolute() and path.is_relative_to(root):
        result = _inside(root, path)
    elif not path.is_absolute():
        result = _inside(root, root / path)
    else:
        # Downloaded runs retain their original /workspace/.../results/<run> paths.
        parts = path.parts
        matches = [i for i in range(len(parts) - 2)
                   if parts[i] == "results" and parts[i + 1] == run_id
                   and parts[i + 2] == category]
        if not matches:
            raise ValueError("History reference is outside this run")
        result = _inside(root, root.joinpath(*parts[matches[-1] + 2:]))
    relative = result.relative_to(root)
    if not relative.parts or relative.parts[0] != category:
        raise ValueError(f"History reference must be under {category}")
    return result


def _declaration(root, path, cutoff, sources, warnings):
    if not path.exists():
        return None
    path = _inside(root, path)
    # mtime is only a conservative exclusion gate, never proof of historical truth.
    if path.stat().st_mtime > cutoff.timestamp():
        warnings.append(f"{path.relative_to(root)}: declaration modified after cutoff; omitted")
        return None
    data = _object(root, path, sources)
    recorded = data.get("recorded_at_utc") or data.get("recorded_at")
    if recorded is not None and _time(recorded) > cutoff:
        warnings.append(f"{path.relative_to(root)}: declaration recorded after cutoff; omitted")
        return None
    from .context import _redact
    return {"data": _redact(data), "provenance": {
        "kind": "caller_declaration", "recorded_at_utc": recorded,
        "temporal_status": "recorded" if recorded else "unverified_file_time_only"}}


def _parent_id(root, value, run_id):
    if value is None:
        return None
    if isinstance(value, str) and _ID.fullmatch(value):
        return value
    path = _reference(root, value, run_id, "trials")
    if len(path.relative_to(root).parts) != 2:
        raise ValueError("History parent must reference a trial directory")
    return path.name


def _report(root, value, run_id, prepared_id, cutoff, sources):
    path = _reference(root, value, run_id, "analogy")
    if len(path.relative_to(root).parts) != 3 or path.name not in {"report.md", "report.json"}:
        raise ValueError("History report must reference analogy/<call>/report.md or report.json")
    manifest = _object(root, path.parent / "manifest.json", sources)
    metadata = manifest.get("input_metadata", {})
    report_run = metadata.get("run_id")
    if ((report_run != run_id and not (manifest.get("stage") == "draft" and report_run is None))
            or metadata.get("prepared_id") != prepared_id):
        raise ValueError("History report belongs to another run or prepared dataset")
    if not manifest.get("finished_at") or _time(manifest["finished_at"]) > cutoff:
        return None
    report = _object(root, path.parent / "report.json", sources)
    fields = ("mechanism_id", "title", "paper_ids", "intervention", "assumptions",
              "limitations", "validation_plan", "rejection_criterion", "evidence_level")
    return {"call_id": path.parent.name, "path": str(path.relative_to(root)),
            "status": manifest.get("status"), "delivery_status": manifest.get("delivery_status"),
            "mechanisms": [{key: m[key] for key in fields if key in m}
                           for m in report.get("mechanisms", []) if isinstance(m, dict)]}


def _selections(root, cutoff, sources, warnings):
    path = root / "results.tsv"
    if not path.exists():
        return []
    path = _inside(root, path)
    if path.stat().st_mtime > cutoff.timestamp():
        warnings.append("results.tsv: undated selection declarations modified after cutoff; omitted")
        return []
    reader = csv.DictReader(io.StringIO(_read(root, path, sources)), delimiter="\t")
    if reader.fieldnames != ["commit", "val_score", "memory_gb", "status", "description"]:
        raise ValueError("History requires the approved five-column results.tsv")
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed history TSV row")
    from .context import _redact
    return _redact(rows)


# 冻结同一运行的历史，严格区分已核验结果与调用者声明。
def build_history(run_dir, *, parent_id, run_id, prepared_id, metric_version, as_of=None):
    from .context import _parent_snapshot

    root = Path(run_dir).expanduser().resolve()
    cutoff = _time(as_of) if as_of is not None else datetime.now(timezone.utc)
    if not isinstance(parent_id, str) or not _ID.fullmatch(parent_id):
        raise ValueError("History parent_id must be a safe trial ID")
    trials = _inside(root, root / "trials")
    if not trials.is_dir():
        raise ValueError("History run directory requires trials/")
    sources, warnings, experiments, nodes = {}, [], [], {}
    for directory in sorted(trials.iterdir()):
        directory = _inside(root, directory)
        if not directory.is_dir():
            continue
        record_path = directory / "source.json"
        if not record_path.exists():
            warnings.append(f"trials/{directory.name}: no source receipt; excluded")
            continue
        record = _object(root, record_path, sources)
        trial_id = record.get("experiment_id")
        if not isinstance(trial_id, str) or not _ID.fullmatch(trial_id) or trial_id != directory.name:
            raise ValueError("History trial ID differs from its directory")
        if record.get("run_id") != run_id or record.get("prepared_id") != prepared_id:
            raise ValueError("History trial belongs to another run or prepared dataset")
        created = _time(record.get("created_at_utc") or record.get("created_at"))
        if created > cutoff:
            sources.pop(str(record_path.relative_to(root)), None)
            continue
        completed_value = record.get("completed_at_utc") or record.get("completed_at")
        completed = _time(completed_value) if completed_value else None
        if completed is not None and completed < created:
            raise ValueError("History completion predates source creation")
        eligible = record.get("execution_status") == "completed" and completed is not None and completed <= cutoff
        item = {"trial_id": trial_id, "parent_trial_id": record.get("parent_experiment_id"),
                "parent_relation_source": "source_receipt" if record.get("parent_experiment_id") else None,
                "created_at_utc": created.isoformat(),
                "completed_at_utc": completed.isoformat() if eligible else None,
                "source_sha256": record.get("sha256"),
                "execution_status": "completed" if eligible else "pending",
                "validation": "verified" if eligible else "not_finalized_as_of_cutoff",
                "adoption": None, "declared_change": None, "rejection": None,
                "selection": None, "report": None}
        # Ensure the validator cannot follow symlinked files outside this run.
        for name in ("source.py", "metrics.json", "config.json") if eligible else ("source.py",):
            _inside(root, directory / name)
        if eligible:
            node, checked, runtime, _, _ = _parent_snapshot(
                directory, prepared_id, metric_version, include_log=False)
            if checked != record:
                raise ValueError("History receipt changed while capturing the snapshot")
            item.update(metrics=runtime["public_validation"], config=runtime["configuration"],
                        score=runtime["public_validation"]["score"],
                        provenance=runtime["provenance"])
            nodes[trial_id] = node
        else:
            source = _read(root, directory / "source.py", sources)
            if hashlib.sha256(source.encode()).hexdigest() != record.get("sha256"):
                raise ValueError("History source hash mismatch")
        adoption = _declaration(root, directory / "adoption.json", cutoff, sources, warnings)
        if adoption is not None:
            data = adoption["data"]
            if data.get("trial_id", trial_id) != trial_id or data.get("run_id", run_id) != run_id:
                raise ValueError("History adoption identity differs from its trial")
            declared_parent = _parent_id(root, data.get("parent"), run_id)
            if item["parent_trial_id"] and declared_parent and item["parent_trial_id"] != declared_parent:
                raise ValueError("History parent declarations disagree")
            item["parent_trial_id"] = item["parent_trial_id"] or declared_parent
            if item["parent_relation_source"] is None and declared_parent:
                item["parent_relation_source"] = "adoption_declaration"
            item["adoption"] = adoption
            item["declared_change"] = data.get("intended_change", data.get("intended_code_change"))
            item["rejection"] = data.get("reason") if data.get("status") == "rejected" else None
            if data.get("report"):
                item["report"] = _report(root, data["report"], run_id, prepared_id, cutoff, sources)
        experiments.append(item)
    by_id = {item["trial_id"]: item for item in experiments}
    if parent_id not in nodes:
        raise ValueError("History parent must be a verified completed trial in this run at cutoff")
    for item in experiments:
        ancestor_id = item["parent_trial_id"]
        if ancestor_id is not None:
            if not isinstance(ancestor_id, str) or not _ID.fullmatch(ancestor_id):
                raise ValueError("History parent trial ID is invalid")
            ancestor = by_id.get(ancestor_id)
            if ancestor and _time(ancestor["created_at_utc"]) >= _time(item["created_at_utc"]):
                raise ValueError("History parent does not precede its child")
            if item["validation"] == "verified" and ancestor and ancestor["validation"] == "verified":
                item["parent_score"] = ancestor["score"]
                item["score_delta"] = item["score"] - ancestor["score"]
                item["score_delta_basis"] = ("Arithmetic comparison of verified scores against the "
                    "caller-declared parent; not independent proof of ancestry or causal effect.")
            elif ancestor is None:
                warnings.append(f"{item['trial_id']}: parent {ancestor_id} unavailable; no score delta")
        if item["trial_id"] in nodes:
            nodes[item["trial_id"]]["parent_id"] = ancestor_id
    selections = _selections(root, cutoff, sources, warnings)
    for row in selections:
        # The TSV commit column is only a source-hash prefix, not a Git commit.
        candidates = [item for item in experiments
                      if isinstance(item["source_sha256"], str)
                      and len(row["commit"]) >= 7 and item["source_sha256"].startswith(row["commit"])]
        named = [item for item in candidates
                 if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(item["trial_id"]) + r"(?![A-Za-z0-9_-])",
                              row["description"])]
        matches = named or candidates
        if len(matches) == 1:
            matches[0]["selection"] = {**row, "provenance": "undated caller declaration; not verified metrics"}
        else:
            warnings.append("results.tsv: ambiguous or unmatched trial row; excluded")
    best_id = None
    best = _declaration(root, root / "best.json", cutoff, sources, warnings)
    if best is not None:
        data = best["data"]
        best_id = data.get("trial_id")
        if data.get("artifact_dir"):
            artifact_id = _parent_id(root, data["artifact_dir"], run_id)
            if best_id is not None and best_id != artifact_id:
                raise ValueError("History best trial ID and artifact path disagree")
            best_id = artifact_id
        if best_id not in nodes or data.get("source_sha256") != by_id[best_id]["source_sha256"]:
            raise ValueError("History best must identify a verified completed source")
    snapshot = {"schema": SCHEMA, "as_of_utc": cutoff.isoformat(), "run_id": run_id,
                "prepared_id": prepared_id, "metric_version": metric_version,
                "parent_trial_id": parent_id, "best_trial_id": best_id,
                "experiments": sorted(experiments, key=lambda item: (item["created_at_utc"], item["trial_id"])),
                "provenance": {"run_dir": str(root), "source_hashes": sources,
                    "pending_note": "Pending means not finalized at cutoff, not proof of a live process.",
                    "declarations_note": "Untimestamped declarations are not proof of historical availability."},
                "warnings": warnings}
    return snapshot, [nodes[item["trial_id"]] for item in snapshot["experiments"]
                      if item["trial_id"] in nodes and item["trial_id"] != parent_id]


# 比较两次冻结历史，供采纳前检查新增结果与父实验变化。
def compare_snapshots(previous, current):
    for key in ("schema", "run_id", "prepared_id", "metric_version"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Cannot compare history with different {key}")
    before = {item["trial_id"]: item for item in previous["experiments"]}
    after = {item["trial_id"]: item for item in current["experiments"]}
    return {"added_trial_ids": sorted(after.keys() - before.keys()),
            "removed_trial_ids": sorted(before.keys() - after.keys()),
            "changed_trial_ids": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
            "parent_changed": previous.get("parent_trial_id") != current.get("parent_trial_id"),
            "previous_parent_trial_id": previous.get("parent_trial_id"),
            "current_parent_trial_id": current.get("parent_trial_id"),
            "best_changed": previous.get("best_trial_id") != current.get("best_trial_id"),
            "previous_best_trial_id": previous.get("best_trial_id"),
            "current_best_trial_id": current.get("best_trial_id")}
