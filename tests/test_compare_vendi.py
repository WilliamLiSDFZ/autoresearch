import builtins
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_vendi as cli


class CompareVendiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "analysis"
        self.cache = self.root / "cache"
        self.input = self.root / "input.jsonl"

    def sample(self, arm, candidate, vector=None, **changes):
        row = {"task": "task", "study_id": "study", "run_id": "run-" + arm,
               "arm": arm, "candidate_id": candidate, "stage": "improve", "view": "implementation",
               "pair_id": "pair-001", "parent_id": "parent", "is_valid": True,
               "text": "A changed loss function connected to the training loop.",
               "assessment_status": "changed", "extraction_status": "ok",
               "representation_version": "diff-v1", "task_hash": "task-hash",
               "prepared_id": "prepared-hash", "evaluator_sha256": "evaluator-hash",
               "metric_version": "metric-v1", "maximize": True}
        if vector is not None:
            row.update(embedding=vector, embedding_model="fixture-vectors-v1")
        row.update(changes)
        return row

    def pair(self, **changes):
        return [self.sample("baseline", "one", [1.0, 0.0], **changes),
                self.sample("baseline", "two", [1.0, 0.0], **changes),
                self.sample("analogy", "one", [1.0, 0.0], **changes),
                self.sample("analogy", "two", [0.0, 1.0], **changes)]

    def write_input(self, rows):
        self.input.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def csv_rows(self, filename):
        with (self.output / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    @contextmanager
    def offline(self):
        original = builtins.__import__
        attempted = []

        def guarded(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"openai", "sentence_transformers", "torch", "matplotlib"}:
                attempted.append(name)
                raise AssertionError("Unexpected model or plotting import: " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            yield
        self.assertEqual(attempted, [], "Offline CLI attempted to import a model or plotting backend")

    def main(self, *extra):
        with self.offline(), redirect_stdout(io.StringIO()):
            return cli.main(["--input", str(self.input), "--out", str(self.output),
                             "--cache", str(self.cache), "--no-plots", "--repeats", "10", *extra])

    def make_raw_run(self):
        run = self.root / "raw" / "run-baseline"
        (run / "trials").mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({
            "run_tag": run.name, "arm": "baseline", "task_id": "task", "study_id": "study",
            "pair_id": "pair-001", "prepared_id": "prepared-hash", "task_file_sha256": "task-hash"}))
        for index in (1, 2):
            trial = run / "trials" / ("trial%04d" % index)
            trial.mkdir()
            source = "value = " + str(index) + "\n"
            (trial / "source.py").write_text(source)
            receipt = {"protocol": "autoresearch-experiment-v1", "run_id": run.name,
                       "experiment_id": trial.name, "prepared_id": "prepared-hash",
                       "execution_status": "pending", "sha256": hashlib.sha256(source.encode()).hexdigest()}
            if index == 1:
                receipt["stage"] = "draft"
            (trial / "source.json").write_text(json.dumps(receipt))
        return run

    def snapshot(self, directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*") if p.is_file()}

    def test_dry_run_writes_nothing_and_loads_no_models(self):
        self.write_input(self.pair())
        before = self.snapshot(self.root)
        self.assertEqual(self.main("--dry-run"), 0)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.cache.exists())

    def test_prepare_only_preserves_missing_parent_coverage_and_raw_files(self):
        run = self.make_raw_run()
        before = self.snapshot(run)
        with self.offline(), redirect_stdout(io.StringIO()):
            code = cli.main(["--runs", str(run), "--out", str(self.output), "--cache", str(self.cache),
                             "--prepare-only", "--no-plots"])
        self.assertEqual(code, 0)
        self.assertEqual(self.snapshot(run), before)
        self.assertFalse(self.cache.exists())
        coverage = self.csv_rows("coverage.csv")
        self.assertEqual(sum(int(row.get("n_candidates") or 0) for row in coverage), 2)
        self.assertEqual(sum(int(row.get("n_missing") or 0) for row in coverage), 1)
        self.assertEqual(self.csv_rows("run_scores.csv"), [])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["summary_api_calls"], 0)
        imported = cli.load_samples(self.output / "samples.jsonl")
        self.assertEqual(len(imported), 2)
        missing_parent = next(row for row in imported if row["candidate_id"] == "trial0002")
        self.assertEqual(missing_parent["extraction_status"], "missing_source")
        self.assertEqual(missing_parent["stage"], "")

    def test_single_candidate_per_run_has_no_score_or_fake_success(self):
        self.write_input([self.sample("baseline", "one", [1, 0]), self.sample("analogy", "one", [0, 1])])
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.csv_rows("run_scores.csv"), [])
        self.assertTrue(all(not row["delta"] for row in self.csv_rows("comparisons.csv")))

    def test_identical_vectors_score_one_orthogonal_score_two_and_pair_delta_one(self):
        self.write_input(self.pair())
        self.assertEqual(self.main(), 0)
        scores = {row["arm"]: row for row in self.csv_rows("run_scores.csv")}
        self.assertAlmostEqual(float(scores["baseline"]["vendi"]), 1.0)
        self.assertAlmostEqual(float(scores["analogy"]["vendi"]), 2.0)
        pair = next(row for row in self.csv_rows("comparisons.csv") if row["kind"] == "pair")
        self.assertAlmostEqual(float(pair["delta"]), 1.0)
        self.assertEqual(pair["n_pairs"], "1")
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["embedding"]["backend"], "precomputed")

    def test_studies_tasks_and_stages_form_separate_cohorts(self):
        rows = []
        for study, task, stage in (("study", "task", "draft"), ("study", "task", "improve"),
                                   ("other-study", "task", "improve"), ("study", "other-task", "improve")):
            for row in self.pair(study_id=study, task=task, stage=stage):
                row["candidate_id"] = stage + "-" + row["candidate_id"]
                row["representation_version"] = "summary-v1" if stage == "draft" else "diff-v1"
                rows.append(row)
        self.write_input(rows)
        self.assertEqual(self.main(), 0)
        scores = self.csv_rows("run_scores.csv")
        self.assertEqual(len(scores), 8)
        self.assertEqual(len({row["cohort"] for row in scores}), 4)
        self.assertTrue(all(row["n_total"] == "2" for row in scores))

    def test_duplicate_pair_arm_runs_are_rejected_without_picking_best(self):
        rows = self.pair()
        rows.append(self.sample("analogy", "third", [0, 1], run_id="another-analogy-run"))
        self.write_input(rows)
        with self.assertRaisesRegex(ValueError, "Multiple runs"):
            self.main()
        self.assertFalse(self.output.exists())

    def test_no_change_and_insufficient_cannot_be_revived_by_saved_vectors(self):
        self.write_input([self.sample("baseline", "one", [1, 0], assessment_status="no_change"),
                          self.sample("baseline", "two", [0, 1], assessment_status="insufficient_evidence")])
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.csv_rows("run_scores.csv"), [])
        coverage = self.csv_rows("coverage.csv")[0]
        self.assertEqual((coverage["n_no_change"], coverage["n_insufficient"], coverage["n_available"]), ("1", "1", "0"))
        exported = cli.load_samples(self.output / "samples.jsonl")
        self.assertTrue(all("embedding" not in row for row in exported))

    def test_valid_only_preserves_excluded_nonscoring_rows(self):
        self.write_input([self.sample("baseline", "one", [1, 0], is_valid=False),
                          self.sample("baseline", "two", [0, 1], is_valid=False, assessment_status="no_change")])
        self.assertEqual(self.main("--valid-only"), 2)
        coverage = self.csv_rows("coverage.csv")[0]
        self.assertEqual(coverage["n_excluded"], "2")
        self.assertEqual(coverage["n_available"], "0")
        self.assertEqual(self.csv_rows("run_scores.csv"), [])

    def test_reembedding_keeps_eligible_text_ready_but_never_revives_exclusions(self):
        import autoresearch_vendi.runtime as runtime
        self.write_input(self.pair() + [self.sample("baseline", "excluded", [0, 1], is_valid=False)])
        observed = []

        def fake_embed(samples, cache, model_name, **kwargs):
            for row in samples:
                observed.append((row["candidate_id"], row.get("extraction_status")))
                if row["candidate_id"] == "excluded":
                    self.assertEqual(row["extraction_status"], "excluded")
                    continue
                self.assertEqual(row.get("extraction_status"), "ok")
                self.assertNotIn("embedding", row)
                row.update(embedding=[1.0, 0.0] if row["arm"] == "baseline" or row["candidate_id"] == "one" else [0.0, 1.0],
                           embedding_model="fake-reembedded-v1")
            return {"backend": "fixture"}

        with patch.object(runtime, "embed_samples", side_effect=fake_embed):
            self.assertEqual(self.main("--valid-only", "--reembed"), 0)
        self.assertEqual(len(observed), 5)
        self.assertEqual(sum(int(row["n_excluded"]) for row in self.csv_rows("coverage.csv")), 1)

    def test_saved_exclusions_survive_input_and_reembedding_without_reapplying_filters(self):
        self.write_input([self.sample("baseline", "one", [1, 0], is_valid=False,
                                      extraction_status="excluded", error="excluded_valid_only"),
                          self.sample("baseline", "two", [0, 1], is_valid=False,
                                      extraction_status="excluded", error="excluded_valid_only")])
        self.assertEqual(self.main("--reembed"), 2)
        self.assertEqual(self.csv_rows("run_scores.csv"), [])
        coverage = self.csv_rows("coverage.csv")[0]
        self.assertEqual(coverage["n_excluded"], "2")
        self.assertEqual(coverage["n_available"], "0")

    def test_empty_input_exits_two_without_fabricated_statistics(self):
        self.write_input([])
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.csv_rows("run_scores.csv"), [])
        self.assertEqual(self.csv_rows("comparisons.csv"), [])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["n_samples"], 0)

    def test_output_cannot_replace_an_original_run(self):
        run = self.make_raw_run()
        before = self.snapshot(run)
        with self.offline(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--runs", str(run), "--out", str(run), "--prepare-only", "--no-plots"])
        self.assertEqual(self.snapshot(run), before)

    def test_output_cannot_overwrite_input_samples_or_reviewed_parent_map(self):
        self.output.mkdir()
        saved_samples = self.output / "samples.jsonl"
        saved_samples.write_text("".join(json.dumps(row) + "\n" for row in self.pair()))
        original_samples = saved_samples.read_bytes()
        with self.offline(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--input", str(saved_samples), "--out", str(self.output), "--prepare-only", "--no-plots"])
        self.assertEqual(saved_samples.read_bytes(), original_samples)

        run = self.make_raw_run()
        parent_map = self.output / "parent_map.csv"
        parent_map.write_text("run,candidate_id,stage,parent_id,child_sha256,parent_sha256,evidence\n")
        original_map = parent_map.read_bytes()
        with self.offline(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--runs", str(run), "--parent-map", str(parent_map), "--out", str(self.output),
                      "--prepare-only", "--no-plots"])
        self.assertEqual(parent_map.read_bytes(), original_map)

    def test_output_and_cache_cannot_be_created_inside_raw_run(self):
        run = self.make_raw_run()
        original = self.snapshot(run)
        for forbidden in ("--out", "--cache"):
            with self.subTest(forbidden=forbidden), self.offline(), redirect_stdout(io.StringIO()), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["--runs", str(run), "--out", str(self.output), "--cache", str(self.cache),
                          forbidden, str(run / "nested-analysis"), "--prepare-only", "--no-plots"])
            self.assertEqual(self.snapshot(run), original)
            self.assertFalse((run / "nested-analysis").exists())

    def test_generated_inventory_input_reports_remedy_before_reading_or_writing(self):
        run = self.make_raw_run()
        with self.offline(), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--runs", str(run), "--out", str(self.output), "--prepare-only"]), 0)
        original = self.snapshot(self.root)
        errors = io.StringIO()
        with self.offline(), redirect_stdout(io.StringIO()), redirect_stderr(errors), self.assertRaises(SystemExit):
            cli.main(["--runs", str(run), "--parent-map", str(self.output / "parent_map.csv"),
                      "--out", str(self.output)])
        self.assertIn("generated parent_map.csv inventory", errors.getvalue())
        self.assertIn("Omit --parent-map", errors.getvalue())
        self.assertEqual(self.snapshot(self.root), original)


if __name__ == "__main__":
    unittest.main()
