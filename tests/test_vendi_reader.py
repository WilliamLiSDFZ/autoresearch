import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoresearch_vendi.reader import MAP_FIELDS, load_candidates


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

    def analogy_evidence(self, run, child, parent="", stage="improve"):
        call_id = stage + "-0001"
        self.json(child / "adoption.json", {"call_id": call_id, "parent": parent})
        if stage == "improve":
            metadata = {"parent_experiment_id": parent, "run_id": self.read(run / "run.json")["run_tag"],
                        "source_sha256": hashlib.sha256((run / "trials" / parent / "source.py").read_bytes()).hexdigest()}
            self.json(run / "analogy" / call_id / "context.json", {"stage": stage, "metadata": metadata})
            self.json(run / "analogy" / call_id / "manifest.json", {"stage": stage, "input_metadata": metadata})
        return run / "analogy" / call_id

    def map_row(self, run, candidate, parent="", stage="improve", **changes):
        row = {"run": run.name, "candidate_id": candidate, "stage": stage, "parent_id": parent,
               "child_sha256": hashlib.sha256((run / "trials" / candidate / "source.py").read_bytes()).hexdigest(),
               "parent_sha256": hashlib.sha256((run / "trials" / parent / "source.py").read_bytes()).hexdigest() if parent else "",
               "evidence": "manually inspected original source and execution notes"}
        row.update(changes)
        return row

    def write_map(self, rows):
        path = self.root / "parents.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MAP_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_completed_and_pending_sources_are_retained_with_separate_validity(self):
        run = self.make_run()
        self.make_trial(run, stage="draft")
        self.make_trial(run, "trial0002", completed=False, stage="improve", parent_id="trial0001")
        samples, issues, parents = load_candidates(self.root)
        self.assertEqual(len(samples), 2)
        self.assertEqual(issues, [])
        self.assertEqual([s["is_valid"] for s in samples], [True, False])
        self.assertEqual([s["status"] for s in samples], ["completed", "pending"])
        self.assertEqual(samples[1]["parent_source"], samples[0]["source"])
        self.assertEqual([s["extraction_status"] for s in samples], ["pending", "pending"])
        self.assertEqual(samples[0]["task"], "actual-task")
        self.assertEqual(samples[1]["representation_version"], "diff-v1")
        self.assertEqual(samples[0]["representation_version"], "summary-v1")
        self.assertTrue(all(p["status"] == "verified" for p in parents))

    def test_invalid_metric_does_not_drop_verified_static_implementation(self):
        run = self.make_run()
        trial = self.make_trial(run, stage="draft")
        (trial / "metrics.json").write_text("{}")
        samples, _, _ = load_candidates(run)
        self.assertFalse(samples[0]["is_valid"])
        self.assertEqual(samples[0]["extraction_status"], "pending")
        self.assertTrue(samples[0]["source"])
        self.assertIn("hash_mismatch", samples[0]["validity_error"])

    def test_source_hash_identity_and_prepared_id_mismatch_block_extraction(self):
        cases = (("sha256", "bad", "source_hash_mismatch"),
                 ("run_id", "other", "source_identity_mismatch"),
                 ("experiment_id", "other", "source_identity_mismatch"),
                 ("prepared_id", "other", "source_prepared_id_mismatch"))
        for index, (field, value, error) in enumerate(cases):
            with self.subTest(field=field):
                run = self.make_run("run" + str(index))
                self.make_trial(run, stage="draft", **{field: value})
                samples, _, _ = load_candidates(run)
                self.assertEqual(samples[0]["extraction_status"], "missing_source")
                self.assertEqual(samples[0]["source"], "")
                self.assertEqual(samples[0]["error"], error)

    def test_missing_source_is_reported_without_losing_candidate_row(self):
        run = self.make_run()
        trial = self.make_trial(run, stage="draft")
        (trial / "source.py").unlink()
        samples, issues, parents = load_candidates(run)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["extraction_status"], "missing_source")
        self.assertEqual(issues[0]["reason"], "no_verified_source_candidates")
        self.assertEqual(parents[0]["status"], "missing")

    def test_previous_number_is_never_invented_as_parent(self):
        run = self.make_run()
        self.make_trial(run, stage="draft")
        self.make_trial(run, "trial0002")
        samples, _, parents = load_candidates(run)
        self.assertEqual(samples[1]["parent_id"], "")
        self.assertEqual(samples[1]["extraction_status"], "missing_source")
        self.assertEqual(samples[1]["error"], "missing_stage_or_parent_evidence")
        self.assertEqual(parents[1]["parent_id"], "")

    def test_initial_tsv_row_identifies_only_the_first_trial_as_draft(self):
        run = self.make_run()
        self.make_trial(run)
        self.make_trial(run, "trial0002")
        self.write(run / "results.tsv", "commit\tval_score\tstatus\tdescription\n"
                   "\t0.8\tkeep\ttrial0001 initial classifier\n"
                   "\t0.7\tdiscard\ttrial0002 initial wording is not a new draft\n")
        samples, _, _ = load_candidates(run)
        self.assertEqual(samples[0]["stage"], "draft")
        self.assertEqual(samples[1]["stage"], "")

    def test_analogy_uses_cross_checked_parent_instead_of_previous_trial(self):
        run = self.make_run(arm="analogy")
        first = self.make_trial(run)
        self.analogy_evidence(run, first, stage="draft")
        self.make_trial(run, "trial0002")
        third = self.make_trial(run, "trial0003")
        self.analogy_evidence(run, third, "trial0001")
        samples, _, parents = load_candidates(run)
        self.assertEqual(samples[2]["parent_id"], "trial0001")
        self.assertEqual(samples[2]["parent_source"], samples[0]["source"])
        self.assertEqual(samples[2]["extraction_status"], "pending")
        self.assertIn("manifest.input_metadata", parents[2]["evidence"])

    def test_analogy_conflicting_or_missing_evidence_blocks_extraction(self):
        for index, broken in enumerate(("parent", "hash", "missing")):
            with self.subTest(broken=broken):
                run = self.make_run("run" + str(index), arm="analogy")
                self.make_trial(run, stage="draft")
                child = self.make_trial(run, "trial0002")
                call = self.analogy_evidence(run, child, "trial0001")
                path = call / "manifest.json"
                if broken == "missing":
                    path.unlink()
                else:
                    payload = self.read(path)
                    payload["input_metadata"]["parent_experiment_id" if broken == "parent" else "source_sha256"] = "wrong"
                    self.json(path, payload)
                samples, _, _ = load_candidates(run)
                self.assertEqual(samples[1]["extraction_status"], "missing_source")

    def test_manual_parent_map_fills_missing_ancestry_and_binds_both_hashes(self):
        run = self.make_run()
        self.make_trial(run)
        self.make_trial(run, "trial0002", completed=False)
        mapping = self.write_map([self.map_row(run, "trial0001", stage="draft"),
                                  self.map_row(run, "trial0002", "trial0001")])
        samples, _, parents = load_candidates(run, parent_map=mapping)
        self.assertEqual([s["stage"] for s in samples], ["draft", "improve"])
        self.assertTrue(all(s["extraction_status"] == "pending" for s in samples))
        self.assertEqual(parents[1]["parent_sha256"], samples[0]["source_hash"])

    def test_parent_map_rejects_unknown_duplicate_incomplete_or_wrong_hash_rows(self):
        run = self.make_run()
        self.make_trial(run, stage="draft")
        self.make_trial(run, "trial0002")
        valid = self.map_row(run, "trial0002", "trial0001")
        cases = ([{**valid, "run": "unknown"}], [{**valid, "candidate_id": "unknown"}],
                 [valid, valid], [{**valid, "evidence": ""}], [{**valid, "parent_sha256": ""}],
                 [{**valid, "child_sha256": "wrong"}], [{**valid, "parent_sha256": "wrong"}],
                 [{**valid, "parent_id": "another-run/trial0001"}])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_candidates(run, parent_map=self.write_map(rows))

    def test_parent_map_cannot_override_recorded_ancestry(self):
        run = self.make_run()
        self.make_trial(run, stage="draft")
        self.make_trial(run, "trial0002", stage="draft")
        self.make_trial(run, "trial0003", parent_id="trial0001")
        mapping = self.write_map([self.map_row(run, "trial0003", "trial0002")])
        with self.assertRaisesRegex(ValueError, "conflicts_with_recorded_ancestry"):
            load_candidates(run, parent_map=mapping)

    def test_unreviewed_inventory_error_identifies_row_candidate_and_remedy(self):
        run = self.make_run()
        self.make_trial(run, stage="draft")
        self.make_trial(run, "trial0002")
        _, _, rows = load_candidates(run)
        mapping = self.write_map([{key: row[key] for key in MAP_FIELDS} for row in rows])
        with self.assertRaises(ValueError) as caught:
            load_candidates(run, parent_map=mapping)
        message = str(caught.exception)
        self.assertIn(str(mapping) + ":3", message)
        self.assertIn("run-baseline/trial0002", message)
        self.assertIn("missing or invalid stage, evidence", message)
        self.assertIn("Omit --parent-map", message)
        self.assertIn("Unresolved candidates remain missing", message)

    def test_self_parent_cycle_and_later_created_parent_are_rejected(self):
        run = self.make_run()
        self.make_trial(run)
        self.make_trial(run, "trial0002")
        with self.assertRaisesRegex(ValueError, "self_parent"):
            load_candidates(run, parent_map=self.write_map([self.map_row(run, "trial0001", "trial0001")]))
        with self.assertRaisesRegex(ValueError, "parent_cycle"):
            load_candidates(run, parent_map=self.write_map([self.map_row(run, "trial0001", "trial0002"),
                                                            self.map_row(run, "trial0002", "trial0001")]))
        for candidate, stamp in (("trial0001", "2026-09-21T02:00:00+00:00"),
                                 ("trial0002", "2026-09-21T01:00:00+00:00")):
            path = run / "trials" / candidate / "source.json"
            receipt = self.read(path)
            receipt["created_at_utc"] = stamp
            self.json(path, receipt)
        with self.assertRaisesRegex(ValueError, "parent_created_after_child"):
            load_candidates(run, parent_map=self.write_map([self.map_row(run, "trial0002", "trial0001")]))

    def test_empty_restart_does_not_make_a_valid_pair_ambiguous(self):
        self.make_run("empty", arm="analogy")
        for name, arm in (("baseline", "baseline"), ("analogy", "analogy")):
            self.make_trial(self.make_run(name, arm=arm), stage="draft")
        samples, issues, _ = load_candidates(self.root)
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(s["pair_id"] == "pair-001" for s in samples))
        self.assertEqual(issues, [{"run": "empty", "reason": "no_verified_source_candidates"}])

    def test_duplicate_source_attempts_block_pairing_even_if_one_is_pending(self):
        self.make_trial(self.make_run("baseline"), stage="draft")
        self.make_trial(self.make_run("analogy", arm="analogy"), stage="draft")
        self.make_trial(self.make_run("analogy-retry", arm="analogy"), completed=False, stage="draft")
        samples, issues, _ = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "" for s in samples))
        self.assertIn("ambiguous_source_runs_per_arm", [issue["reason"] for issue in issues])
        self.assertEqual(len(samples), 3)

    def test_incompatible_prepared_data_blocks_pairing_but_preserves_sources(self):
        self.make_trial(self.make_run("baseline"), stage="draft")
        self.make_trial(self.make_run("analogy", arm="analogy", prepared_id="other-split"), stage="draft")
        samples, issues, _ = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "" and s["source"] for s in samples))
        self.assertIn("prepared_id", issues[0]["reason"])

    def test_manifest_exclusion_and_unknown_run_are_respected(self):
        self.make_trial(self.make_run(), stage="draft")
        manifest = self.write(self.root / "manifest.csv", "run,exclude_reason\nrun-baseline,debug\n")
        samples, issues, _ = load_candidates(self.root, manifest=manifest)
        self.assertEqual(samples, [])
        self.assertEqual(issues[0]["reason"], "manual:debug")
        self.write(manifest, "run,arm\nmissing,baseline\n")
        with self.assertRaises(ValueError):
            load_candidates(self.root, manifest=manifest)

    def test_missing_pair_identity_keeps_source_but_disables_pairing(self):
        self.make_trial(self.make_run("orphan", pair_id="", study_id=""), stage="draft")
        samples, issues, _ = load_candidates(self.root)
        self.assertEqual(samples[0]["pair_id"], "")
        self.assertEqual(samples[0]["extraction_status"], "pending")
        self.assertEqual(issues[0]["reason"], "missing_pair_identity_use_manifest")

    def test_source_symlink_is_not_followed(self):
        run = self.make_run()
        trial = self.make_trial(run, stage="draft")
        outside = self.write(self.root / "outside.py", "value = 42\n")
        (trial / "source.py").unlink()
        (trial / "source.py").symlink_to(outside)
        samples, _, _ = load_candidates(run)
        self.assertEqual(samples[0]["source"], "")
        self.assertEqual(samples[0]["extraction_status"], "missing_source")
        self.assertIn("symlink", samples[0]["error"])

    def test_nonexistent_result_root_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "existing directory"):
            load_candidates(self.root / "missing")

    def test_unknown_metric_in_failed_run_does_not_disable_static_pairing(self):
        self.make_trial(self.make_run("baseline"), stage="draft")
        self.make_trial(self.make_run("analogy", arm="analogy"), completed=False, stage="draft")
        samples, issues, _ = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "pair-001" for s in samples))
        self.assertEqual({issue["reason"] for issue in issues},
                         {"pair_metric_version_unknown", "pair_maximize_unknown"})

    def test_known_metric_conflict_disables_static_pairing(self):
        self.make_trial(self.make_run("baseline"), stage="draft")
        trial = self.make_trial(self.make_run("analogy", arm="analogy"), stage="draft")
        path = trial / "metrics.json"
        metrics = self.read(path)
        metrics["maximize"] = False
        self.json(path, metrics)
        receipt = self.read(trial / "source.json")
        receipt["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.json(trial / "source.json", receipt)
        samples, issues, _ = load_candidates(self.root)
        self.assertTrue(all(s["pair_id"] == "" for s in samples))
        self.assertEqual(issues[0]["reason"], "incompatible_pair:maximize")

    def test_solutions_ignore_all_ancestry_and_keep_only_neutral_source_references(self):
        run = self.make_run(arm="analogy")
        first = self.make_trial(run, stage="invalid", parent_id="missing")
        child = self.make_trial(run, "trial0002", parent_id="trial0002")
        self.write(first / "adoption.json", "not json")
        self.write(child / "adoption.json", "not json")
        self.write(run / "results.tsv", "not a trial inventory")
        with patch("autoresearch_vendi.reader._ancestry", side_effect=AssertionError("must not read ancestry")), \
                patch("autoresearch_vendi.reader._parent_map", side_effect=AssertionError("must not read parent map")):
            samples, issues, parents = load_candidates(run, mode="solutions")
        self.assertEqual(issues, [])
        self.assertEqual(parents, [])
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample["stage"], "solution")
            self.assertEqual(sample["view"], "implementation")
            self.assertEqual(sample["representation_version"], "solution-v1")
            self.assertEqual(sample["extraction_status"], "pending")
            self.assertEqual(sample["parent_id"], "")
            self.assertEqual(sample["parent_source"], "")
            self.assertNotIn("parent_evidence", sample)
            self.assertEqual(sample["source_refs"], [str(run / "trials" / sample["candidate_id"] / name)
                                                    for name in ("source.py", "source.json")])
        changes, _, _ = load_candidates(run)
        self.assertTrue(all(sample["extraction_status"] == "missing_source" for sample in changes))

    def test_solutions_preserve_failed_pending_and_invalid_completed_sources(self):
        run = self.make_run()
        self.make_trial(run)
        self.make_trial(run, "trial0002", completed=False)
        self.make_trial(run, "trial0003", execution_status="failed")
        bad_metric = self.make_trial(run, "trial0004")
        self.write(bad_metric / "metrics.json", "{}")
        samples, _, parents = load_candidates(run, mode="solutions")
        self.assertEqual([sample["status"] for sample in samples], ["completed", "pending", "failed", "completed"])
        self.assertEqual([sample["is_valid"] for sample in samples], [True, False, False, False])
        self.assertTrue(all(sample["extraction_status"] == "pending" for sample in samples))
        self.assertTrue(all(sample["metric_version"] == "jubias-continuous-auc-v1" for sample in samples))
        self.assertIn("hash_mismatch", samples[-1]["validity_error"])
        self.assertEqual(parents, [])

    def test_solutions_reject_parent_map_before_reading_it_and_invalid_modes(self):
        run = self.make_run()
        with self.assertRaisesRegex(ValueError, "does not accept a parent map"):
            load_candidates(run, parent_map=self.root / "nonexistent.csv", mode="solutions")
        with self.assertRaisesRegex(ValueError, "mode must be changes or solutions"):
            load_candidates(run, mode="unknown")

    def test_solutions_do_not_extract_bytes_from_unverified_receipts(self):
        cases = (("sha256", "bad", "source_hash_mismatch"),
                 ("protocol", "unknown", "unknown_source_protocol"),
                 ("run_id", "other", "source_identity_mismatch"),
                 ("experiment_id", "other", "source_identity_mismatch"),
                 ("prepared_id", "other", "source_prepared_id_mismatch"))
        for index, (field, value, error) in enumerate(cases):
            with self.subTest(field=field):
                run = self.make_run("run" + str(index))
                self.make_trial(run, **{field: value})
                samples, _, parents = load_candidates(run, mode="solutions")
                self.assertTrue(samples[0]["source_hash"])
                self.assertEqual(samples[0]["source"], "")
                self.assertEqual(samples[0]["extraction_status"], "missing_source")
                self.assertEqual(samples[0]["stage"], "solution")
                self.assertEqual(samples[0]["error"], error)
                self.assertEqual(parents, [])

    def test_solutions_preserve_identical_sources_as_distinct_candidates(self):
        run = self.make_run()
        first = self.make_trial(run)
        second = self.make_trial(run, "trial0002")
        self.write(second / "source.py", (first / "source.py").read_text())
        receipt = self.read(second / "source.json")
        receipt["sha256"] = hashlib.sha256((second / "source.py").read_bytes()).hexdigest()
        self.json(second / "source.json", receipt)
        samples, _, _ = load_candidates(run, mode="solutions")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["source_hash"], samples[1]["source_hash"])
        self.assertEqual([sample["candidate_id"] for sample in samples], ["trial0001", "trial0002"])

    def test_solutions_do_not_follow_source_receipt_or_trial_symlinks(self):
        for index, target in enumerate(("source.py", "source.json", "trial")):
            with self.subTest(target=target):
                run = self.make_run("run" + str(index))
                trial = self.make_trial(run)
                path = trial if target == "trial" else trial / target
                outside = self.root / ("outside" + str(index))
                path.rename(outside)
                path.symlink_to(outside)
                samples, _, _ = load_candidates(run, mode="solutions")
                self.assertEqual(samples[0]["source"], "")
                self.assertEqual(samples[0]["extraction_status"], "missing_source")
                self.assertIn("symlink", samples[0]["error"])

    def test_solutions_retain_pair_safety_checks(self):
        self.make_trial(self.make_run("baseline"))
        self.make_trial(self.make_run("analogy", arm="analogy", prepared_id="other-split"))
        samples, issues, _ = load_candidates(self.root, mode="solutions")
        self.assertTrue(all(sample["pair_id"] == "" and sample["source"] for sample in samples))
        self.assertEqual(issues[0]["reason"], "incompatible_pair:prepared_id")
        self.make_trial(self.make_run("analogy-retry", arm="analogy", prepared_id="other-split"), completed=False)
        samples, issues, _ = load_candidates(self.root, mode="solutions")
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample["pair_id"] == "" for sample in samples))
        self.assertIn("ambiguous_source_runs_per_arm", [issue["reason"] for issue in issues])


if __name__ == "__main__":
    unittest.main()
