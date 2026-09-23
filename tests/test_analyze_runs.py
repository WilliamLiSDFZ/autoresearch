import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import analyze_runs as analysis


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class AnalyzeRunsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def write_json(self, path, payload):
        return self.write(path, json.dumps(payload))

    def read_json(self, path):
        return json.loads(path.read_text())

    def make_run(self, name, arm="baseline", scores=(0.8,), maximize=True, **metadata):
        run = self.root / name
        values = {
            "run_tag": name, "arm": arm, "task_id": "jigsaw", "study_id": "study",
            "pair_id": "pair-001", "task_file_sha256": "a" * 64,
            "prepared_id": "b" * 64, "seed": 1337, "gpu": "fixture GPU",
            "start_commit": "c" * 40, "resources": "1 GPU / 8 CPU / 32Gi",
            "budget_seconds": 21600, "agent_model": "fixture-agent",
        }
        values.update(metadata)
        self.write_json(run / "run.json", values)
        self.write(run / "summary.md", "Completed fixture experiment.\n")
        self.write(run / "_worktree/prepare.py", "# fixed evaluator\n")
        self.write(run / "_worktree/uv.lock", "# fixed environment\n")
        for index, score in enumerate(scores, 1):
            self.make_trial(run, "trial%04d" % index, score, maximize=maximize)
        return run

    def make_trial(self, run, trial_id, score, maximize=True, completed=True, **metrics):
        metadata = self.read_json(run / "run.json")
        trial = run / "trials" / trial_id
        self.write(trial / "source.py", "# frozen candidate " + trial_id + "\n")
        self.write_json(trial / "config.json", {"seed": metadata.get("seed", 1337)})
        self.write(trial / "run.log", "Finished fixture training.\n")
        self.write(trial / "validation_predictions.npy", b"\x93NUMPY-fixture")
        payload = {
            "score": score, "maximize": maximize, "metric_version": "jubias-continuous-auc-v1",
            "prepared_id": metadata["prepared_id"], "validation_rows": 100,
        }
        payload.update(metrics)
        self.write_json(trial / "metrics.json", payload)
        receipt = {
            "protocol": "autoresearch-experiment-v1", "run_id": metadata["run_tag"],
            "experiment_id": trial_id, "prepared_id": metadata["prepared_id"],
            "execution_status": "completed" if completed else "pending",
            "source_path": "/remote/worktree/train.py",
        }
        for filename, field in (("source.py", "sha256"), ("config.json", "config_sha256"),
                                ("metrics.json", "metrics_sha256"), ("run.log", "log_sha256")):
            receipt[field] = hashlib.sha256((trial / filename).read_bytes()).hexdigest()
        self.write_json(trial / "source.json", receipt)
        return trial

    def make_pair(self, pair_id="pair-001", baseline=(0.7,), analogy=(0.8,), **metadata):
        self.make_run(pair_id + "-baseline", scores=baseline, pair_id=pair_id, **metadata)
        self.make_run(pair_id + "-analogy", arm="analogy", scores=analogy, pair_id=pair_id, **metadata)

    def test_inventory_uses_verified_metrics_and_checks_best_pointer(self):
        run = self.make_run("valid", scores=(0.700001, 0.8123456789))
        self.write_json(run / "best.json", {"trial_id": "trial0001", "score": 999,
                                           "source_sha256": "wrong"})
        self.write(run / "results.tsv", "commit\tval_score\nFAKE\t999\n")
        records, trials = analysis.inventory_runs(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "eligible")
        self.assertEqual(records[0]["trial"], "trial0002")
        self.assertEqual(records[0]["score"], 0.8123456789)
        self.assertEqual(records[0]["n_valid_trials"], 2)
        self.assertIn("best_pointer_differs_from_verified_optimum", records[0]["warnings"])
        self.assertTrue(all(trial["status"] == "eligible" for trial in trials))
        self.assertEqual(analysis.inventory_runs(run)[0][0]["run"], "valid")

    def test_empty_pending_nonfinite_and_boolean_scores_are_not_results(self):
        self.make_run("empty", scores=())
        pending = self.make_run("pending", scores=())
        self.make_trial(pending, "trial0001", 0.9, completed=False)
        for name, score in (("nan", float("nan")), ("inf", float("inf")),
                            ("negative-inf", float("-inf")), ("bool", True)):
            self.make_run(name, scores=(score,))
        records, trials = analysis.inventory_runs(self.root)
        self.assertTrue(all(row["status"] == "excluded" for row in records))
        self.assertTrue(all(row["reason"] == "no_valid_completed_result" for row in records))
        reasons = {row["run"]: row["reason"] for row in trials}
        self.assertEqual(reasons["pending"], "not_completed")
        self.assertEqual(reasons["nan"], "missing_or_nonfinite_score")
        self.assertEqual(reasons["bool"], "missing_or_nonfinite_score")
        self.assertEqual(analysis.build_pairs(records), ([], []))

    def test_completed_artifact_tampering_and_missing_predictions_are_rejected(self):
        for filename in ("source.py", "config.json", "metrics.json", "run.log"):
            run = self.make_run("tamper-" + filename)
            path = run / "trials/trial0001" / filename
            path.write_bytes(path.read_bytes() + b" ")
        run = self.make_run("missing-predictions")
        (run / "trials/trial0001/validation_predictions.npy").unlink()
        records, trials = analysis.inventory_runs(self.root)
        self.assertTrue(all(row["status"] == "excluded" for row in records))
        reasons = {row["run"]: row["reason"] for row in trials}
        for filename in ("source.py", "config.json", "metrics.json", "run.log"):
            self.assertEqual(reasons["tamper-" + filename], "hash_mismatch:" + filename)
        self.assertEqual(reasons["missing-predictions"], "missing_validation_predictions")

    def test_receipt_protocol_identity_and_prepared_id_must_match(self):
        cases = (("protocol", "unknown", "unknown_receipt_protocol"),
                 ("run_id", "another-run", "receipt_identity_mismatch"),
                 ("experiment_id", "another-trial", "receipt_identity_mismatch"),
                 ("prepared_id", "another-split", "prepared_id_mismatch"))
        for field, value, expected in cases:
            run = self.make_run(field)
            path = run / "trials/trial0001/source.json"
            receipt = self.read_json(path)
            receipt[field] = value
            self.write_json(path, receipt)
            _, trials = analysis.inventory_runs(run)
            self.assertEqual(trials[0]["reason"], expected)

    def test_inconsistent_metrics_within_one_run_excludes_the_run(self):
        run = self.make_run("mixed")
        self.make_trial(run, "trial0002", 0.9, metric_version="different-metric")
        records, trials = analysis.inventory_runs(self.root)
        self.assertEqual(records[0]["reason"], "inconsistent_trial_metrics")
        self.assertEqual(records[0]["status"], "excluded")
        self.assertEqual(len(trials), 2)

    def test_missing_metric_direction_is_not_inferred_from_score(self):
        run = self.make_run("missing-direction", scores=())
        self.make_trial(run, "trial0001", 0.8, maximize=None)
        _, trials = analysis.inventory_runs(self.root)
        self.assertEqual(trials[0]["reason"], "missing_metric_identity_or_direction")

    def test_minimize_selects_smallest_and_positive_effect_means_improvement(self):
        self.make_pair(baseline=(4.0, 2.0), analogy=(3.0, 1.0), maximize=False)
        records, _ = analysis.inventory_runs(self.root)
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(issues, [])
        self.assertEqual(pairs[0]["reference_score"], 2.0)
        self.assertEqual(pairs[0]["treatment_score"], 1.0)
        self.assertEqual(pairs[0]["effect"], 1.0)
        reversed_pairs, _ = analysis.build_pairs(records, contrasts=[("baseline", "analogy")])
        self.assertEqual(reversed_pairs[0]["effect"], -1.0)

    def test_maximize_effect_and_single_pair_has_no_interval(self):
        self.make_pair(baseline=(0.6, 0.7), analogy=(0.75, 0.8))
        records, _ = analysis.inventory_runs(self.root)
        pairs, issues = analysis.build_pairs(records)
        summaries = analysis.summarize_effects(pairs)
        self.assertEqual(issues, [])
        self.assertAlmostEqual(pairs[0]["effect"], 0.1)
        self.assertEqual(summaries[0]["n_pairs"], 1)
        self.assertIsNone(summaries[0]["ci95_low"])
        self.assertIsNone(summaries[0]["ci95_high"])

    def test_pair_requires_same_task_study_split_metric_and_evaluator(self):
        self.make_pair()
        records, _ = analysis.inventory_runs(self.root)
        changes = {"task_id": "another-task", "study_id": "another-study",
                   "prepared_id": "another-split", "metric_version": "another-metric",
                   "maximize": False, "validation_rows": 101, "task_hash": "another-description",
                   "evaluator_sha256": "another-evaluator"}
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(records)
                changed[0][field] = value
                pairs, issues = analysis.build_pairs(changed)
                self.assertEqual(pairs, [])
                self.assertTrue(issues)

    def test_task_hash_alias_pairs_with_canonical_metadata(self):
        self.make_pair()
        path = self.root / "pair-001-analogy/run.json"
        metadata = self.read_json(path)
        metadata["task_sha256"] = metadata.pop("task_file_sha256")
        for canonical in (None, "", "a" * 64):
            with self.subTest(canonical=canonical):
                values = dict(metadata)
                if canonical is not None:
                    values["task_file_sha256"] = canonical
                self.write_json(path, values)
                records, _ = analysis.inventory_runs(self.root)
                pairs, issues = analysis.build_pairs(records)
                self.assertEqual(issues, [])
                self.assertEqual(len(pairs), 1)
                self.assertTrue(all(row["task_hash"] == "a" * 64 for row in records))

    def test_conflicting_task_hash_aliases_exclude_the_run(self):
        run = self.make_run("conflicting-task", task_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "conflicting_task_hashes"):
            analysis.identify_run(run, self.read_json(run / "run.json"), {})
        records, _ = analysis.inventory_runs(self.root)
        self.assertEqual(records[0]["status"], "excluded")
        self.assertEqual(records[0]["reason"], "invalid_run_metadata:conflicting_task_hashes")

    def test_gpu_metadata_aliases_preserve_precedence_and_pairing(self):
        self.make_pair()
        path = self.root / "pair-001-analogy/run.json"
        metadata = self.read_json(path)
        metadata.pop("gpu")
        cases = (
            {"gpu_name": "fixture GPU"},
            {"gpu": "", "gpu_name": "fixture GPU"},
            {"gpu_actual": "fixture GPU", "gpu_name": "ignored GPU"},
            {"gpu": "fixture GPU", "gpu_actual": "ignored GPU", "gpu_name": "ignored GPU"},
        )
        for aliases in cases:
            with self.subTest(aliases=aliases):
                self.write_json(path, dict(metadata, **aliases))
                records, _ = analysis.inventory_runs(self.root)
                pairs, issues = analysis.build_pairs(records)
                self.assertTrue(all(row["gpu"] == "fixture GPU" for row in records))
                self.assertEqual(issues, [])
                self.assertNotIn("gpu_differs", pairs[0]["warnings"])
                self.assertNotIn("gpu_unknown", pairs[0]["warnings"])

    def test_duplicate_eligible_arm_is_ambiguous_even_if_one_scores_higher(self):
        self.make_pair()
        self.make_run("another-analogy-attempt", arm="analogy", scores=(0.99,))
        records, _ = analysis.inventory_runs(self.root)
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(pairs, [])
        self.assertEqual(issues[0]["reason"], "missing_or_ambiguous_arm")
        self.assertIn("another-analogy-attempt", issues[0]["runs"])

    def test_explicit_exclusions_pair_reruns_without_selecting_highest_attempt(self):
        self.make_pair(baseline=(0.7,), analogy=(0.8,))
        for arm, score in (("baseline", 0.98), ("analogy", 0.99)):
            run = self.make_run("interrupted-" + arm, arm=arm, scores=(score,))
            (run / "summary.md").unlink()
            self.make_trial(run, "trial0002", 1.0, completed=False)
            self.write_json(run / "_fetch.json", {"fetched_at_utc": "2099-01-01T00:00:00Z"})
        records, _ = analysis.inventory_runs(self.root)
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(pairs, [])
        self.assertEqual(issues[0]["reason"], "missing_or_ambiguous_arm")

        manifest = self.root / "manifest.csv"
        self.write(manifest, "run,exclude_reason\n"
                             "interrupted-baseline,infrastructure interruption\n"
                             "interrupted-analogy,infrastructure interruption\n")
        records, trials = analysis.inventory_runs(self.root, analysis.load_overrides(manifest))
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(issues, [])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["reference_run"], "pair-001-baseline")
        self.assertEqual(pairs[0]["treatment_run"], "pair-001-analogy")
        self.assertAlmostEqual(pairs[0]["effect"], 0.1)
        for row in records:
            if row["run"].startswith("interrupted-"):
                self.assertEqual(row["status"], "excluded")
                self.assertEqual(row["reason"], "manual:infrastructure interruption")
                self.assertEqual(row["n_valid_trials"], 1)
        self.assertEqual(len(trials), 6)

    def test_failed_restart_does_not_hide_the_only_valid_attempt(self):
        self.make_pair()
        self.make_run("failed-analogy-attempt", arm="analogy", scores=())
        records, _ = analysis.inventory_runs(self.root)
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(issues, [])
        self.assertEqual(pairs[0]["treatment_run"], "pair-001-analogy")

    def test_missing_pair_identity_requires_manifest(self):
        self.make_run("orphan", pair_id="", study_id="")
        records, _ = analysis.inventory_runs(self.root)
        self.assertEqual(records[0]["status"], "eligible")
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(pairs, [])
        self.assertEqual(issues, [{"run": "orphan", "reason": "missing_identity_use_manifest"}])

    def test_changed_hardware_is_a_warning_and_not_hidden(self):
        self.make_pair()
        records, _ = analysis.inventory_runs(self.root)
        records[0]["gpu"] = "another GPU"
        records[0]["start_commit"] = "another-commit"
        records[0]["budget_seconds"] = ""
        pairs, issues = analysis.build_pairs(records)
        self.assertEqual(issues, [])
        self.assertIn("gpu_differs", pairs[0]["warnings"])
        self.assertIn("start_commit_differs", pairs[0]["warnings"])
        self.assertIn("budget_seconds_unknown", pairs[0]["warnings"])

    def test_custom_arms_and_manifest_overrides_do_not_change_measured_scores(self):
        self.make_run("control", arm="control")
        self.make_run("retrieval-v2", arm="retrieval-v2", scores=(0.9,))
        self.make_run("excluded", arm="retrieval-v2", scores=(0.99,))
        manifest = self.root / "manifest.csv"
        self.write(manifest, "run,task_id,study_id,pair_id,arm,exclude_reason\n"
                            "control,new-task,new-study,draw-A,none,\n"
                            "retrieval-v2,new-task,new-study,draw-A,kb,\n"
                            "excluded,,,,,debug attempt\n")
        overrides = analysis.load_overrides(manifest)
        records, _ = analysis.inventory_runs(self.root, overrides)
        pairs, issues = analysis.build_pairs(records, reference="none", contrasts=[("kb", "none")])
        self.assertEqual(issues, [])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["treatment"], "kb")
        self.assertEqual(pairs[0]["reference"], "none")
        self.assertEqual(pairs[0]["task_id"], "new-task")
        self.assertAlmostEqual(pairs[0]["effect"], 0.1)
        self.assertEqual(next(r for r in records if r["run"] == "excluded")["reason"], "manual:debug attempt")

    def test_manifest_rejects_score_overrides_duplicates_and_unknown_runs(self):
        manifest = self.root / "manifest.csv"
        for contents in ("run,score\na,1.0\n", "run,arm\na,baseline\na,analogy\n",
                         "arm\nbaseline\n", "run,arm\n,baseline\n"):
            with self.subTest(contents=contents):
                self.write(manifest, contents)
                with self.assertRaises(ValueError):
                    analysis.load_overrides(manifest)
        with self.assertRaisesRegex(ValueError, "missing runs"):
            analysis.inventory_runs(self.root, {"unknown": {"arm": "baseline"}})
        self.assertEqual(analysis.load_overrides(None), {})

    @unittest.skipUnless(HAS_SCIPY, "paired confidence intervals require scipy")
    def test_independent_pairs_across_seeds_count_pairs_not_candidates(self):
        self.make_pair("pair-001", baseline=(0.6, 0.7), analogy=(0.75, 0.8), seed=1)
        self.make_pair("pair-002", baseline=(0.6, 0.65), analogy=(0.7, 0.95),
                       seed=2, prepared_id="different-seed-split")
        records, trials = analysis.inventory_runs(self.root)
        pairs, issues = analysis.build_pairs(records)
        summaries = analysis.summarize_effects(pairs)
        self.assertEqual(issues, [])
        self.assertEqual(len(trials), 8)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["n_pairs"], 2)
        self.assertAlmostEqual(summaries[0]["mean_effect"], 0.2)
        self.assertAlmostEqual(summaries[0]["ci95_low"], -1.0706204736174704)
        self.assertAlmostEqual(summaries[0]["ci95_high"], 1.4706204736174704)

    def test_effect_cohorts_do_not_pool_distinct_tasks_studies_or_metrics(self):
        self.make_pair()
        records, _ = analysis.inventory_runs(self.root)
        pairs, _ = analysis.build_pairs(records)
        for field, value in (("task_id", "other-task"), ("study_id", "other-study"),
                             ("metric_version", "other-metric"), ("evaluator_sha256", "other-code")):
            with self.subTest(field=field):
                changed = copy.deepcopy(records)
                for row in changed:
                    row[field] = value
                    row["pair_id"] = "pair-002"
                new_pairs, issues = analysis.build_pairs(changed)
                self.assertEqual(issues, [])
                summaries = analysis.summarize_effects(pairs + new_pairs)
                self.assertEqual(len(summaries), 2)
                self.assertTrue(all(row["n_pairs"] == 1 for row in summaries))


if __name__ == "__main__":
    unittest.main()
