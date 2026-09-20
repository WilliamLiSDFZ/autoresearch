"""Offline provenance/context checks: no training code, datasets, or API calls."""
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest

from autoresearch_analogy.code_tools import CodeReadingSession, CodeToolOptions
from autoresearch_analogy.context import ContextOptions, build_packet, runtime_evidence_paths
from autoresearch_analogy import report_v2


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha(path.read_bytes())


class PacketFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="analogy-context-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "description.md"
        self.task.write_text("# Jigsaw\nClassify comments using the fixed subgroup-weighted AUC.\n")
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        manifest = {
            "version": 1, "task_id": "jigsaw-unintended-bias-in-toxicity-classification",
            "metric_version": "jubias-continuous-auc-v1", "maximize": True,
            "seed": 42, "validation_fraction": 0.05,
            "split_method": "autoresearch-numpy-stratified-target-v1",
            "train_rows": 950, "validation_rows": 50, "test_rows": 100,
        }
        manifest["prepared_id"] = sha(json.dumps(manifest, sort_keys=True, allow_nan=False).encode())
        self.prepared_id = manifest["prepared_id"]
        write_json(self.prepared / "manifest.json", manifest)
        self.parent = self.root / "exp001"
        self.parent.mkdir()
        self.source = "raise RuntimeError('candidate must never execute')\n\ndef update(value):\n    weight = 2\n    return value * weight\n"
        (self.parent / "source.py").write_text(self.source)
        self.metrics = {"score": 0.8, "overall_auc": 0.9,
            "power_means": {"subgroup": 0.8, "bpsn": 0.75, "bnsp": 0.75},
            "identities": {"male": {"subgroup": {"auc": 0.7, "rows": 10,
                "positive_count": 4, "negative_count": 6}}},
            "prepared_id": self.prepared_id, "metric_version": "jubias-continuous-auc-v1",
            "maximize": True, "validation_rows": 50}
        self.record = {"sha256": sha(self.source.encode()), "experiment_id": "exp001",
            "run_id": "run_001", "prepared_id": self.prepared_id, "execution_status": "completed",
            "metrics_sha256": write_json(self.parent / "metrics.json", self.metrics),
            "config_sha256": write_json(self.parent / "config.json", {"learning_rate": 0.01})}
        self.store_record()

    def store_record(self):
        write_json(self.parent / "source.json", self.record)

    def build(self, **kwargs):
        return build_packet("improve", self.task, self.prepared, self.parent, **kwargs)


class BuildPacketTests(PacketFixture):
    def test_draft_uses_metadata_without_datasets_or_parent(self):
        text, metadata, runtime, session = build_packet("draft", self.task, self.prepared)
        self.assertIsNone(session)
        self.assertEqual(metadata["stage"], "draft")
        self.assertEqual(metadata["prepared_id"], self.prepared_id)
        self.assertNotIn("public_validation", runtime)
        self.assertNotIn("exp001", text)
        # There are deliberately no CSV files in this fixture.
        self.assertEqual(list(self.prepared.iterdir()), [self.prepared / "manifest.json"])
        with self.assertRaisesRegex(ValueError, "Draft"):
            build_packet("draft", self.task, self.prepared, self.parent)
        with self.assertRaisesRegex(ValueError, "Draft"):
            build_packet("draft", self.task, self.prepared, history=self.root / "results.tsv")

    def test_completed_source_is_frozen_and_only_read_lines_are_evidence(self):
        text, metadata, runtime, session = self.build(resources="one assigned GPU")
        self.assertEqual(metadata["parent_experiment_id"], "exp001")
        self.assertEqual(metadata["run_id"], "run_001")
        self.assertEqual(runtime["public_validation"]["score"], 0.8)
        self.assertIn("caller-supplied", runtime["resources"]["source"])
        anchor = {"node_id": "exp001", "source_sha256": self.record["sha256"],
                  "start_line": 4, "end_line": 5}
        self.assertFalse(session.validate_anchor(anchor))
        result = session.dispatch("read_candidate_code", {"node_id": "exp001", "symbol": "update"})
        self.assertEqual(result["status"], "ok")
        self.assertTrue(session.validate_anchor(anchor))
        self.assertIn("weight = 2", result["content"])
        # Later edits cannot change the episode's captured evidence.
        (self.parent / "source.py").write_text("different_source = True\n")
        frozen = session.dispatch("read_candidate_code", {"node_id": "exp001", "symbol": "update"})
        self.assertIn("weight = 2", frozen["content"])
        self.assertNotIn("different_source", frozen["content"])
        self.assertTrue(report_v2._runtime_path("public_validation.power_means.subgroup", runtime))
        self.assertFalse(report_v2._runtime_path("private_test.score", runtime))

    def test_wrong_hash_prepared_id_pending_and_modified_metric_are_rejected(self):
        for key, value, reason in (("sha256", "0" * 64, "source hash"),
                                  ("prepared_id", "wrong", "prepared_id"),
                                  ("execution_status", "pending", "completed")):
            with self.subTest(key=key):
                original = self.record[key]
                self.record[key] = value
                self.store_record()
                with self.assertRaisesRegex(ValueError, reason):
                    self.build()
                self.record[key] = original
                self.store_record()
        self.metrics["score"] = 0.85
        write_json(self.parent / "metrics.json", self.metrics)
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            self.build()

    def test_metric_identity_and_missing_artifact_binding_are_rejected(self):
        self.record.pop("config_sha256")
        self.store_record()
        with self.assertRaisesRegex(ValueError, "artifact hashes"):
            self.build()
        self.record["config_sha256"] = sha((self.parent / "config.json").read_bytes())
        self.metrics["prepared_id"] = "another-preparation"
        self.record["metrics_sha256"] = write_json(self.parent / "metrics.json", self.metrics)
        self.store_record()
        with self.assertRaisesRegex(ValueError, "different prepared"):
            self.build()

    def test_unbound_log_is_not_read_and_bound_log_is_cleaned(self):
        log = self.parent / "run.log"
        log.write_text("SHOULD_NOT_ENTER_CONTEXT\n")
        self.assertNotIn("SHOULD_NOT_ENTER_CONTEXT", self.build()[0])
        log.write_text("training observed\nFutureWarning: deprecation\nloss=0.2\n")
        self.record["log_sha256"] = sha(log.read_bytes())
        self.store_record()
        text, _, _, _ = self.build()
        self.assertIn("training observed", text)
        self.assertNotIn("FutureWarning", text)

    def test_history_is_explicit_bounded_and_not_runtime_evidence(self):
        history = self.root / "results.tsv"
        history.write_text("commit\tval_score\tmemory_gb\tstatus\tdescription\n"
                           + "".join(f"c{i}\t0.7\t8\tdiscard\tclaim-{i}\n" for i in range(20)))
        text, metadata, runtime, _ = self.build(history=history, options=ContextOptions(trajectory_nodes=3))
        self.assertEqual(metadata["history"]["rows_total"], 20)
        self.assertEqual(metadata["history"]["rows_selected"], 3)
        self.assertIn("claim-19", text)
        self.assertNotIn("claim-16", text)
        self.assertNotIn("history", runtime)

    def test_omitted_runtime_has_no_legal_evidence_path(self):
        _, metadata, runtime, _ = self.build(options=ContextOptions(runtime_chars=600))
        paths = runtime_evidence_paths(runtime)
        self.assertEqual(paths, metadata["runtime_evidence_paths"])
        self.assertFalse(report_v2._runtime_path("public_validation.score", runtime))
        self.assertTrue(metadata["runtime_omitted"])

    def test_tampered_prepared_manifest_is_rejected(self):
        manifest = json.loads((self.prepared / "manifest.json").read_text())
        manifest["train_rows"] += 1
        write_json(self.prepared / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            self.build()

    def test_sensitive_config_fields_are_redacted(self):
        self.record["config_sha256"] = write_json(self.parent / "config.json",
            {"api_key": "fake-sensitive-value", "nested": {"password": "do-not-send"}, "learning_rate": 0.01})
        self.store_record()
        text, _, runtime, _ = self.build()
        self.assertNotIn("fake-sensitive-value", text)
        self.assertNotIn("do-not-send", text)
        self.assertEqual(runtime["configuration"]["learning_rate"], 0.01)

    def test_explicit_same_run_reference_enables_real_source_diff(self):
        reference = self.root / "exp000"
        shutil.copytree(self.parent, reference)
        source = self.source.replace("weight = 2", "weight = 1")
        (reference / "source.py").write_text(source)
        ref_record = {**self.record, "experiment_id": "exp000", "sha256": sha(source.encode())}
        write_json(reference / "source.json", ref_record)
        _, metadata, runtime, session = self.build(reference_artifacts=[reference])
        self.assertEqual({node["node_id"] for node in session.allowed_nodes()}, {"exp000", "exp001"})
        self.assertEqual(metadata["reference_artifacts"][0]["experiment_id"], "exp000")
        result = session.dispatch("diff_candidate_code", {"base_node_id": "exp000", "target_node_id": "exp001"})
        self.assertEqual(result["status"], "ok")
        self.assertIn("-    weight = 1", result["content"])
        self.assertIn("+    weight = 2", result["content"])
        self.assertEqual(runtime["reference_candidates"][0]["candidate"]["experiment_id"], "exp000")
        with self.assertRaisesRegex(ValueError, "Draft"):
            build_packet("draft", self.task, self.prepared, reference_artifacts=[reference])
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.build(reference_artifacts=[reference, reference])
        ref_record["run_id"] = "another-run"
        write_json(reference / "source.json", ref_record)
        with self.assertRaisesRegex(ValueError, "run_id"):
            self.build(reference_artifacts=[reference])
        with self.assertRaisesRegex(ValueError, "trajectory_nodes"):
            self.build(reference_artifacts=[reference, reference], options=ContextOptions(trajectory_nodes=1))


class CodeEvidenceTests(unittest.TestCase):
    def test_partial_line_and_unknown_node_do_not_grant_citations(self):
        source = "def update():\n    return '" + "x" * 10000 + "'\n"
        node = {"id": "exp", "code": source, "execution_status": "completed"}
        session = CodeReadingSession([node], "exp", options=CodeToolOptions(max_read_chars=2500))
        result = session.dispatch("read_candidate_code", {"start_line": 2, "end_line": 2})
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["truncated"])
        ref = {"node_id": "exp", "source_sha256": sha(source.encode()), "start_line": 2, "end_line": 2}
        self.assertFalse(session.validate_anchor(ref))
        self.assertEqual(session.dispatch("read_candidate_code", {"node_id": "unapproved"})["status"], "error")

    def test_runtime_path_schema_and_read_anchor_validation(self):
        runtime = {"public_validation": {"score": 0.9}, "missing": None,
                   "observations": [{"count": 2}], "unknown": "unknown"}
        self.assertTrue(report_v2._runtime_path("observations.0.count", runtime))
        for path in ("observations[0].count", "missing", "unknown", "public_validation.hidden"):
            self.assertFalse(report_v2._runtime_path(path, runtime))
        issues = report_v2._anchor_issues(
            [{"node_id": "x", "source_sha256": "a" * 64, "start_line": 1, "end_line": 3}],
            "facts.0", None)
        self.assertEqual(issues[0]["code"], "code_anchor_unread")

    def test_full_report_cannot_claim_unread_code_or_unavailable_runtime(self):
        base = {"report_schema_revision": 2, "bottlenecks": [], "mechanisms": [],
                "abstention_reason": "Insufficient evidence", "hypotheses": [], "unknowns": []}
        fact = {"statement": "Validation score was measured", "source": "runtime",
                "evidence": "The displayed metric", "runtime_evidence": ["public_validation.score"]}
        result = report_v2.validate_detailed({**base, "observed_facts": [fact]}, set(), object(), 3,
            reading=None, abstracts={}, code_session=None, runtime_context={}, mode="improve")
        self.assertFalse(result["shared_facts_valid"])
        self.assertIn("runtime_path_unavailable", [issue["code"] for issue in result["issues"]])
        result = report_v2.validate_detailed({**base, "observed_facts": [fact]}, set(), object(), 3,
            reading=None, abstracts={}, code_session=None,
            runtime_context={"public_validation": {"score": 0.8}}, mode="improve")
        self.assertTrue(result["shared_facts_valid"])
        self.assertFalse(result["issues"])


if __name__ == "__main__":
    unittest.main()
