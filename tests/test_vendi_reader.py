import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from autoresearch_vendi.reader import load_candidates


class VendiReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def json(self, path, content):
        return self.write(path, json.dumps(content))

    def read(self, path):
        return json.loads(path.read_text())

    def make_run(self, name="run-baseline", **extra):
        run = self.root / name
        metadata = {"run_tag": name, "arm": "baseline", "task_id": "actual-task",
                    "study_id": "study", "pair_id": "pair-001", "task_file_sha256": "task-hash",
                    "prepared_id": "prepared-hash", "seed": 42}
        metadata.update(extra)
        self.json(run / "run.json", metadata)
        self.write(run / "_worktree/prepare.py", "# frozen evaluator\n")
        return run

    def make_trial(self, run, candidate="trial0001", completed=True, **receipt_extra):
        trial = run / "trials" / candidate
        metadata = self.read(run / "run.json")
        self.write(trial / "source.py", "# " + candidate + "\nvalue = 1\n")
        self.json(trial / "metrics.json", {"score": 0.8, "maximize": True,
                                          "metric_version": "jubias-continuous-auc-v1",
                                          "prepared_id": metadata["prepared_id"], "validation_rows": 10})
        self.json(trial / "config.json", {"seed": 42})
        self.write(trial / "validation_predictions.npy", "fixture-predictions")
        receipt = {"protocol": "autoresearch-experiment-v1", "run_id": metadata["run_tag"],
                   "experiment_id": candidate, "prepared_id": metadata["prepared_id"],
                   "execution_status": "completed" if completed else "pending"}
        for filename, field in (("source.py", "sha256"), ("metrics.json", "metrics_sha256"),
                                ("config.json", "config_sha256")):
            receipt[field] = hashlib.sha256((trial / filename).read_bytes()).hexdigest()
        receipt.update(receipt_extra)
        self.json(trial / "source.json", receipt)
        return trial

    def test_invalid_metric_does_not_drop_verified_static_implementation(self):
        run = self.make_run()
        trial = self.make_trial(run)
        (trial / "metrics.json").write_text("{}")
        samples, _ = load_candidates(run)
        self.assertFalse(samples[0]["is_valid"])
        self.assertEqual(samples[0]["extraction_status"], "pending")
        self.assertTrue(samples[0]["source"])
        self.assertIn("hash_mismatch", samples[0]["validity_error"])

    def test_missing_source_is_reported_without_losing_candidate_row(self):
        run = self.make_run()
        trial = self.make_trial(run)
        (trial / "source.py").unlink()
        samples, issues = load_candidates(run)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["extraction_status"], "missing_source")
        self.assertEqual(issues[0]["reason"], "no_verified_source_candidates")

    def test_empty_restart_does_not_make_a_valid_pair_ambiguous(self):
        self.make_run("empty", arm="analogy")
        for name, arm in (("baseline", "baseline"), ("analogy", "analogy")):
            self.make_trial(self.make_run(name, arm=arm))
        samples, issues = load_candidates(self.root)
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(s["pair_id"] == "pair-001" for s in samples))
        self.assertEqual(issues, [{"run": "empty", "reason": "no_verified_source_candidates"}])

    def test_duplicate_source_attempts_block_pairing_even_if_one_is_pending(self):
        self.make_trial(self.make_run("baseline"))
        self.make_trial(self.make_run("analogy", arm="analogy"))
        self.make_trial(self.make_run("analogy-retry", arm="analogy"), completed=False)
        samples, issues = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "" for s in samples))
        self.assertIn("ambiguous_source_runs_per_arm", [issue["reason"] for issue in issues])
        self.assertEqual(len(samples), 3)

    def test_incompatible_artifact_identities_block_pairing_but_preserve_sources(self):
        cases = (("prepared_id", "prepared_id"), ("task_file_sha256", "task_hash"),
                 ("evaluator", "evaluator_sha256"))
        for index, (field, issue_field) in enumerate(cases):
            with self.subTest(field=field):
                pair = "pair-" + str(index)
                self.make_trial(self.make_run("baseline" + str(index), pair_id=pair))
                extra = {field: "other-identity"} if field != "evaluator" else {}
                run = self.make_run("analogy" + str(index), arm="analogy", pair_id=pair, **extra)
                if field == "evaluator":
                    self.write(run / "_worktree/prepare.py", "# different evaluator\n")
                self.make_trial(run)
                samples, issues = load_candidates(self.root)
                self.assertTrue(all(s["pair_id"] == "" and s["source"] for s in samples))
                self.assertIn("incompatible_pair:" + issue_field, [issue["reason"] for issue in issues])

    def test_manifest_exclusion_and_unknown_run_are_respected(self):
        self.make_trial(self.make_run())
        manifest = self.write(self.root / "manifest.csv", "run,exclude_reason\nrun-baseline,debug\n")
        samples, issues = load_candidates(self.root, manifest=manifest)
        self.assertEqual(samples, [])
        self.assertEqual(issues[0]["reason"], "manual:debug")
        self.write(manifest, "run,arm\nmissing,baseline\n")
        with self.assertRaises(ValueError):
            load_candidates(self.root, manifest=manifest)

    def test_missing_pair_identity_keeps_source_but_disables_pairing(self):
        self.make_trial(self.make_run("orphan", pair_id="", study_id=""))
        samples, issues = load_candidates(self.root)
        self.assertEqual(samples[0]["pair_id"], "")
        self.assertEqual(samples[0]["extraction_status"], "pending")
        self.assertEqual(issues[0]["reason"], "missing_pair_identity_use_manifest")

    def test_manifest_can_supply_missing_pair_identity(self):
        self.make_trial(self.make_run("orphan", pair_id="", study_id=""))
        manifest = self.write(self.root / "manifest.csv", "run,study_id,pair_id,arm\n"
                              "orphan,reviewed-study,reviewed-pair,analogy\n")
        samples, issues = load_candidates(self.root, manifest=manifest)
        self.assertEqual(issues, [])
        self.assertEqual(samples[0]["study_id"], "reviewed-study")
        self.assertEqual(samples[0]["pair_id"], "reviewed-pair")
        self.assertEqual(samples[0]["arm"], "analogy")

    def test_nonexistent_result_root_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "existing directory"):
            load_candidates(self.root / "missing")

    def test_unknown_metric_in_failed_run_does_not_disable_static_pairing(self):
        self.make_trial(self.make_run("baseline"))
        self.make_trial(self.make_run("analogy", arm="analogy"), completed=False)
        samples, issues = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "pair-001" for s in samples))
        self.assertEqual({issue["reason"] for issue in issues},
                         {"pair_metric_version_unknown", "pair_maximize_unknown"})

    def test_known_metric_conflict_disables_static_pairing(self):
        self.make_trial(self.make_run("baseline"))
        trial = self.make_trial(self.make_run("analogy", arm="analogy"))
        path = trial / "metrics.json"
        metrics = self.read(path)
        metrics["maximize"] = False
        self.json(path, metrics)
        receipt = self.read(trial / "source.json")
        receipt["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.json(trial / "source.json", receipt)
        samples, issues = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "" for s in samples))
        self.assertEqual(issues[0]["reason"], "incompatible_pair:maximize")

    def test_ignore_all_ancestry_and_keep_only_neutral_source_references(self):
        run = self.make_run(arm="analogy")
        first = self.make_trial(run, stage="invalid", parent_id="missing")
        child = self.make_trial(run, "trial0002", parent_id="trial0002")
        self.write(first / "adoption.json", "not json")
        self.write(child / "adoption.json", "not json")
        self.write(run / "results.tsv", "not a trial inventory")
        samples, issues = load_candidates(run)
        self.assertEqual(issues, [])
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample["stage"], "solution")
            self.assertEqual(sample["view"], "implementation")
            self.assertEqual(sample["representation_version"], "solution-v1")
            self.assertEqual(sample["extraction_status"], "pending")
            self.assertNotIn("parent_id", sample)
            self.assertNotIn("parent_source", sample)
            self.assertNotIn("parent_evidence", sample)
            self.assertEqual(sample["source_refs"], [str(run / "trials" / sample["candidate_id"] / name)
                                                    for name in ("source.py", "source.json")])

    def test_solution_preserves_full_source_hash_and_run_metadata(self):
        run = self.make_run(gpu="fixture-gpu", start_commit="abc123", budget_seconds=300, agent_model="fixture-model")
        trial = self.make_trial(run)
        source = "# complete solution\r\nvalue = '完整方案'\r\n"
        (trial / "source.py").write_bytes(source.encode("utf-8"))
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        receipt = self.read(trial / "source.json")
        receipt["sha256"] = source_hash
        self.json(trial / "source.json", receipt)
        samples, issues = load_candidates(run)
        self.assertEqual(issues, [])
        self.assertEqual(samples[0]["source"], source)
        self.assertEqual(samples[0]["source_hash"], source_hash)
        expected = {"task": "actual-task", "prepared_id": "prepared-hash", "task_hash": "task-hash",
                    "gpu": "fixture-gpu", "start_commit": "abc123", "seed": 42, "budget_seconds": 300,
                    "agent_model": "fixture-model", "evaluator_sha256":
                    hashlib.sha256((run / "_worktree/prepare.py").read_bytes()).hexdigest()}
        self.assertEqual({field: samples[0][field] for field in expected}, expected)

    def test_inconsistent_metric_identity_within_run_disables_pairing(self):
        run = self.make_run()
        self.make_trial(run)
        second = self.make_trial(run, "trial0002")
        metrics = self.read(second / "metrics.json")
        metrics["metric_version"] = "other-metric"
        path = self.json(second / "metrics.json", metrics)
        receipt = self.read(second / "source.json")
        receipt["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.json(second / "source.json", receipt)
        samples, issues = load_candidates(run)
        self.assertTrue(all(sample["is_valid"] and sample["source"] and not sample["pair_id"] for sample in samples))
        self.assertEqual(issues, [{"run": run.name, "reason": "inconsistent_run_metric_identity"}])

    def test_preserve_failed_pending_and_invalid_completed_sources(self):
        run = self.make_run()
        self.make_trial(run)
        self.make_trial(run, "trial0002", completed=False)
        self.make_trial(run, "trial0003", execution_status="failed")
        bad_metric = self.make_trial(run, "trial0004")
        self.write(bad_metric / "metrics.json", "{}")
        samples, _ = load_candidates(run)
        self.assertEqual([sample["status"] for sample in samples], ["completed", "pending", "failed", "completed"])
        self.assertEqual([sample["is_valid"] for sample in samples], [True, False, False, False])
        self.assertTrue(all(sample["extraction_status"] == "pending" for sample in samples))
        self.assertTrue(all(sample["metric_version"] == "jubias-continuous-auc-v1" for sample in samples))
        self.assertIn("hash_mismatch", samples[-1]["validity_error"])

    def test_do_not_extract_bytes_from_unverified_receipts(self):
        cases = (("sha256", "bad", "source_hash_mismatch"),
                 ("protocol", "unknown", "unknown_source_protocol"),
                 ("run_id", "other", "source_identity_mismatch"),
                 ("experiment_id", "other", "source_identity_mismatch"),
                 ("prepared_id", "other", "source_prepared_id_mismatch"))
        for index, (field, value, error) in enumerate(cases):
            with self.subTest(field=field):
                run = self.make_run("run" + str(index))
                self.make_trial(run, **{field: value})
                samples, _ = load_candidates(run)
                self.assertTrue(samples[0]["source_hash"])
                self.assertEqual(samples[0]["source"], "")
                self.assertEqual(samples[0]["extraction_status"], "missing_source")
                self.assertEqual(samples[0]["stage"], "solution")
                self.assertEqual(samples[0]["error"], error)

    def test_preserve_identical_sources_as_distinct_candidates(self):
        run = self.make_run()
        first = self.make_trial(run)
        second = self.make_trial(run, "trial0002")
        self.write(second / "source.py", (first / "source.py").read_text())
        receipt = self.read(second / "source.json")
        receipt["sha256"] = hashlib.sha256((second / "source.py").read_bytes()).hexdigest()
        self.json(second / "source.json", receipt)
        samples, _ = load_candidates(run)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["source_hash"], samples[1]["source_hash"])
        self.assertEqual([sample["candidate_id"] for sample in samples], ["trial0001", "trial0002"])

    def test_do_not_follow_source_receipt_or_trial_symlinks(self):
        for index, target in enumerate(("source.py", "source.json", "trial")):
            with self.subTest(target=target):
                run = self.make_run("run" + str(index))
                trial = self.make_trial(run)
                path = trial if target == "trial" else trial / target
                outside = self.root / ("outside" + str(index))
                path.rename(outside)
                path.symlink_to(outside)
                samples, _ = load_candidates(run)
                self.assertEqual(samples[0]["source"], "")
                self.assertEqual(samples[0]["extraction_status"], "missing_source")
                self.assertIn("symlink", samples[0]["error"])


if __name__ == "__main__":
    unittest.main()
