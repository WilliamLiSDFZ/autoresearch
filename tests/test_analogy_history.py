"""Offline history integrity and prefetch chronology checks."""
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoresearch_analogy.history import build_history, compare_snapshots


def digest(content):
    return hashlib.sha256(content).hexdigest()


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="analogy-history-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run_001"
        (self.root / "trials").mkdir(parents=True)
        self.prepared = "a" * 64
        self.cutoff = "2026-01-02T12:00:00+00:00"
        self.cutoff_seconds = datetime.fromisoformat(self.cutoff).timestamp()
        self.make_trial("trial0001", score=0.8)

    def write(self, path, value, *, future=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(value).encode()
        path.write_bytes(content)
        stamp = self.cutoff_seconds + 1 if future else self.cutoff_seconds - 1
        os.utime(path, (stamp, stamp))
        return digest(content)

    def make_trial(self, tid, *, score=None, parent=None, created="2026-01-01T00:00:00+00:00",
                   completed="2026-01-02T00:00:00+00:00"):
        directory = self.root / "trials" / tid
        directory.mkdir()
        source = f"raise RuntimeError('never execute {tid}')\n".encode()
        (directory / "source.py").write_bytes(source)
        record = {"experiment_id": tid, "run_id": "run_001", "prepared_id": self.prepared,
                  "sha256": digest(source), "created_at_utc": created,
                  "execution_status": "pending"}
        if score is not None:
            record.update(execution_status="completed", completed_at_utc=completed)
            metrics = {"score": score, "prepared_id": self.prepared,
                       "metric_version": "test-metric", "maximize": True,
                       "overall_auc": score + 0.02, "power_means": {"subgroup": score}}
            record["metrics_sha256"] = self.write(directory / "metrics.json", metrics)
            record["config_sha256"] = self.write(directory / "config.json", {"weight": 0.05})
            # A deliberately absent log must not be loaded, even though the receipt binds it.
            record["log_sha256"] = "b" * 64
        self.write(directory / "source.json", record)
        if parent:
            self.write(directory / "adoption.json", {"trial_id": tid, "parent": parent,
                       "status": "rejected", "reason": "Report rejected; independent trial",
                       "intended_change": "weight 0.05 -> 0.02"})
        return record

    def build(self, **kwargs):
        return build_history(self.root, parent_id=kwargs.pop("parent_id", "trial0001"),
                             run_id="run_001", prepared_id=self.prepared,
                             metric_version="test-metric", as_of=kwargs.pop("as_of", self.cutoff), **kwargs)

    def test_completed_history_has_actual_parent_delta_and_separate_adoption_status(self):
        self.make_trial("trial0002", score=0.85, parent="trial0001", created="2026-01-01T01:00:00Z")
        self.make_trial("trial0003", score=0.84, parent="trial0002", created="2026-01-01T02:00:00Z")
        snapshot, nodes = self.build()
        child = snapshot["experiments"][2]
        self.assertEqual(child["parent_trial_id"], "trial0002")
        self.assertAlmostEqual(child["score_delta"], -0.01)
        self.assertEqual(child["execution_status"], "completed")
        self.assertEqual(child["adoption"]["data"]["status"], "rejected")
        self.assertEqual([node["id"] for node in nodes], ["trial0002", "trial0003"])
        self.assertEqual(nodes[1]["parent_id"], "trial0002")
        json.dumps(snapshot, allow_nan=False)

    def test_prefetch_cutoff_omits_future_metrics_adoption_and_new_trials(self):
        self.make_trial("trial0002", score=0.1, parent="trial0001",
                        created="2026-01-02T11:00:00Z", completed="2026-01-02T13:00:00Z")
        self.make_trial("trial0003", score=0.9, created="2026-01-02T14:00:00Z",
                        completed="2026-01-02T15:00:00Z")
        adoption = self.root / "trials/trial0002/adoption.json"
        self.write(adoption, {"reason": "FUTURE_RESULT"}, future=True)
        original = Path.read_bytes
        def guarded(path):
            if path.parent.name == "trial0002" and path.name in {"metrics.json", "config.json", "adoption.json"}:
                self.fail(f"Read future contents: {path}")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            snapshot, nodes = self.build()
        self.assertEqual([x["trial_id"] for x in snapshot["experiments"]], ["trial0001", "trial0002"])
        pending = snapshot["experiments"][1]
        self.assertEqual(pending["execution_status"], "pending")
        self.assertNotIn("score", pending)
        self.assertIsNone(pending["adoption"])
        self.assertEqual(nodes, [])
        self.assertNotIn("FUTURE_RESULT", json.dumps(snapshot))
        self.assertNotIn("trial0003", json.dumps(snapshot))

    def test_missing_completion_time_never_exposes_score(self):
        self.make_trial("trial0002", score=0.7)
        path = self.root / "trials/trial0002/source.json"
        record = json.loads(path.read_text())
        record.pop("completed_at_utc")
        self.write(path, record)
        snapshot, nodes = self.build()
        self.assertEqual(snapshot["experiments"][1]["execution_status"], "pending")
        self.assertNotIn("score", snapshot["experiments"][1])
        self.assertEqual(nodes, [])

    def test_wrong_run_prepared_source_metric_and_paths_fail(self):
        record_path = self.root / "trials/trial0001/source.json"
        original = json.loads(record_path.read_text())
        for key, value in [("run_id", "other_run"), ("prepared_id", "wrong"), ("sha256", "0" * 64)]:
            with self.subTest(key=key):
                self.write(record_path, {**original, key: value})
                with self.assertRaises(ValueError):
                    self.build()
        self.write(record_path, original)
        self.write(self.root / "trials/trial0001/metrics.json", {"score": 0.99})
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            self.build()

    def test_symlink_escape_rejected_without_reading_target(self):
        external = Path(self.temp.name) / "outside"
        external.mkdir()
        (self.root / "trials/foreign").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.build()

    def test_metric_symlink_escape_and_foreign_parent_rejected(self):
        metric = self.root / "trials/trial0001/metrics.json"
        outside = Path(self.temp.name) / "external-metrics.json"
        metric.replace(outside)
        metric.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.build()
        metric.unlink()
        outside.replace(metric)
        self.write(self.root / "trials/trial0001/adoption.json", {
            "parent": "/workspace/results/other_run/trials/trial0000"})
        with self.assertRaisesRegex(ValueError, "outside this run"):
            self.build()

    def test_timestamped_future_declaration_omitted_even_with_old_mtime(self):
        self.write(self.root / "trials/trial0001/adoption.json", {
            "recorded_at_utc": "2026-01-02T13:00:00Z", "reason": "future outcome"})
        snapshot, _ = self.build()
        self.assertIsNone(snapshot["experiments"][0]["adoption"])
        self.assertNotIn("future outcome", json.dumps(snapshot))

    def make_report(self, *, run_id="run_001", stage="improve"):
        directory = self.root / "analogy/improve-0001"
        metadata = {"prepared_id": self.prepared}
        if stage != "draft":
            metadata["run_id"] = run_id
        self.write(directory / "manifest.json", {"stage": stage, "input_metadata": metadata,
                   "status": "ok", "finished_at": "2026-01-02T11:30:00Z"})
        self.write(directory / "report.json", {"mechanisms": [{"mechanism_id": "m1",
                   "title": "real mechanism", "intervention": "A" * 20000}]})
        self.write(self.root / "trials/trial0001/adoption.json", {"trial_id": "trial0001",
                   "report": "/workspace/project/results/run_001/analogy/improve-0001/report.md"})

    def test_explicit_relocated_same_run_report_has_no_character_truncation(self):
        self.make_report()
        snapshot, _ = self.build()
        report = snapshot["experiments"][0]["report"]
        self.assertEqual(report["mechanisms"][0]["intervention"], "A" * 20000)
        self.make_report(stage="draft")
        self.assertIsNotNone(self.build()[0]["experiments"][0]["report"])

    def test_cross_run_report_path_and_manifest_fail(self):
        self.make_report(run_id="other_run")
        with self.assertRaisesRegex(ValueError, "another run"):
            self.build()
        self.write(self.root / "trials/trial0001/adoption.json", {
            "report": "/workspace/project/results/other_run/analogy/improve-0001/report.md"})
        with self.assertRaisesRegex(ValueError, "outside this run"):
            self.build()

    def test_tsv_selection_is_declaration_and_does_not_mark_pending_completed(self):
        record = self.make_trial("trial0002")
        path = self.root / "results.tsv"
        path.write_text("commit\tval_score\tmemory_gb\tstatus\tdescription\n"
                        f"{record['sha256'][:7]}\t0.99\t8\tcrash\ttrial0002 interrupted\n")
        os.utime(path, (self.cutoff_seconds - 1, self.cutoff_seconds - 1))
        pending = self.build()[0]["experiments"][1]
        self.assertEqual(pending["selection"]["status"], "crash")
        self.assertEqual(pending["execution_status"], "pending")
        self.assertNotIn("score", pending)
        os.utime(path, (self.cutoff_seconds + 1, self.cutoff_seconds + 1))
        self.assertIsNone(self.build()[0]["experiments"][1]["selection"])

    def test_snapshot_comparison_detects_new_completion_and_best_changes(self):
        self.make_trial("trial0002", created="2026-01-01T01:00:00Z")
        before, _ = self.build()
        after = copy.deepcopy(before)
        after["experiments"][1]["execution_status"] = "completed"
        after["experiments"].append({"trial_id": "trial0003"})
        after["parent_trial_id"] = after["best_trial_id"] = "trial0002"
        changes = compare_snapshots(before, after)
        self.assertEqual(changes["changed_trial_ids"], ["trial0002"])
        self.assertEqual(changes["added_trial_ids"], ["trial0003"])
        self.assertTrue(changes["parent_changed"])
        self.assertTrue(changes["best_changed"])
        after["run_id"] = "other"
        with self.assertRaises(ValueError):
            compare_snapshots(before, after)

    def test_best_requires_matching_verified_source(self):
        record = json.loads((self.root / "trials/trial0001/source.json").read_text())
        best = {"trial_id": "trial0001", "source_sha256": record["sha256"],
                "artifact_dir": "/workspace/results/run_001/trials/trial0001"}
        self.write(self.root / "best.json", best)
        self.assertEqual(self.build()[0]["best_trial_id"], "trial0001")
        self.write(self.root / "best.json", {**best, "source_sha256": "bad"})
        with self.assertRaisesRegex(ValueError, "verified completed"):
            self.build()


if __name__ == "__main__":
    unittest.main()
