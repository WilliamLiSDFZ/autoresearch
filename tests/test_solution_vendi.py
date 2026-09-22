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

import compare_solution_vendi as cli


class SolutionVendiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.output = self.root / "analysis"
        self.cache = self.root / "cache"
        self.input = self.root / "input.jsonl"

    def make_run(self, arm="baseline"):
        run = self.raw / ("run-" + arm)
        run.mkdir(parents=True)
        metadata = {"run_tag": run.name, "arm": arm, "task_id": "task", "study_id": "study",
                    "pair_id": "pair-001", "prepared_id": "prepared-hash", "task_file_sha256": "task-hash",
                    "unused_score": "PERFORMANCE-METADATA-SENTINEL"}
        (run / "run.json").write_text(json.dumps(metadata))
        (run / "results.tsv").write_text("RETRIEVAL-NARRATIVE-SENTINEL")
        for index in (1, 2):
            trial = run / "trials" / ("trial%04d" % index)
            trial.mkdir(parents=True)
            source = "value = " + str(index) + "\n"
            (trial / "source.py").write_text(source)
            receipt = {"protocol": "autoresearch-experiment-v1", "run_id": run.name,
                       "experiment_id": trial.name, "prepared_id": "prepared-hash",
                       "execution_status": "pending", "sha256": hashlib.sha256(source.encode()).hexdigest()}
            if index == 1:
                receipt["stage"] = "draft"
            (trial / "source.json").write_text(json.dumps(receipt))
            (trial / "adoption.json").write_text("INVALID-ANALOGY-REPORT-SENTINEL")
        return run

    def sample(self, arm, candidate, vector, **changes):
        row = {"task": "task", "study_id": "study", "run_id": "run-" + arm,
               "arm": arm, "candidate_id": candidate, "stage": "solution", "view": "implementation",
               "pair_id": "pair-001", "is_valid": True, "text": "A source-grounded method description.",
               "extraction_status": "ok", "representation_version": "solution-v1",
               "task_hash": "task-hash", "prepared_id": "prepared-hash", "evaluator_sha256": "evaluator-hash",
               "embedding": vector, "embedding_model": "fixture-vectors-v1"}
        row.update(changes)
        return row

    def pair(self, **changes):
        return [self.sample("baseline", "one", [1, 0], **changes),
                self.sample("baseline", "two", [1, 0], **changes),
                self.sample("analogy", "one", [1, 0], **changes),
                self.sample("analogy", "two", [0, 1], **changes)]

    def write_input(self, rows):
        self.input.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def csv_rows(self, filename):
        with (self.output / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    @contextmanager
    def offline(self, stdlib=False):
        original = builtins.__import__
        forbidden = {"openai", "sentence_transformers", "torch", "matplotlib"}
        if stdlib:
            forbidden.add("numpy")
        attempted = []

        def guarded(name, *args, **kwargs):
            if name.split(".", 1)[0] in forbidden:
                attempted.append(name)
                raise AssertionError("Unexpected backend import: " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            yield
        self.assertEqual(attempted, [])

    def main(self, *extra, raw=False, stdlib=False):
        source = ["--runs", str(self.raw)] if raw else ["--input", str(self.input)]
        with self.offline(stdlib=stdlib), redirect_stdout(io.StringIO()):
            return cli.main([*source, "--out", str(self.output), "--cache", str(self.cache),
                             "--no-plots", *extra])

    def test_dry_run_reads_all_sources_without_ancestry_models_or_writes(self):
        self.make_run()
        self.make_run("analogy")
        before = self.snapshot(self.root)
        self.assertEqual(self.main("--dry-run", raw=True, stdlib=True), 0)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.cache.exists())

    def test_prepare_only_lists_whole_solutions_and_emits_no_parent_inventory(self):
        self.make_run()
        before = self.snapshot(self.raw)
        self.assertEqual(self.main("--prepare-only", raw=True, stdlib=True), 0)
        self.assertEqual(self.snapshot(self.raw), before)
        self.assertFalse(self.cache.exists())
        self.assertFalse((self.output / "parent_map.csv").exists())
        self.assertFalse((self.output / "change_evidence").exists())
        self.assertEqual(len(self.csv_rows("solution_cards.csv")), 2)
        self.assertEqual(self.csv_rows("full_run_scores.csv"), [])
        coverage = self.csv_rows("coverage.csv")
        self.assertEqual((coverage[0]["stage"], coverage[0]["n_candidates"], coverage[0]["n_pending"]),
                         ("solution", "2", "2"))
        self.assertEqual(coverage[0]["n_missing"], "0")
        self.assertNotIn("n_changed", coverage[0])
        exported = [json.loads(line) for line in (self.output / "samples.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["stage"] == "solution" and row["parent_id"] == "" for row in exported))
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["summary_api_calls"], 0)
        self.assertEqual(manifest["settings"]["mode"], "solutions")
        self.assertEqual(manifest["settings"]["embedding_backend"], "openai")
        self.assertEqual(manifest["settings"]["embedding_model"], "text-embedding-3-small")

    def test_precomputed_vectors_keep_duplicates_and_produce_expected_matched_delta(self):
        import autoresearch_vendi.runtime as runtime
        self.write_input(self.pair())
        original = self.input.read_bytes()
        with patch.object(runtime, "Summarizer", side_effect=AssertionError("No extraction for saved vectors")):
            self.assertEqual(self.main(), 0)
        self.assertEqual(self.input.read_bytes(), original)
        scores = {row["arm"]: row for row in self.csv_rows("run_scores.csv")}
        self.assertAlmostEqual(float(scores["baseline"]["vendi"]), 1.0)
        self.assertAlmostEqual(float(scores["analogy"]["vendi"]), 2.0)
        self.assertEqual(scores["baseline"]["n_total"], "2")
        pair = next(row for row in self.csv_rows("comparisons.csv") if row["kind"] == "pair")
        self.assertAlmostEqual(float(pair["delta"]), 1.0)
        self.assertEqual(len(self.csv_rows("solution_cards.csv")), 4)
        self.assertEqual(len(self.csv_rows("full_run_scores.csv")), 2)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["embedding"]["backend"], "precomputed")
        self.assertEqual(manifest["summary_api_calls"], 0)
        self.assertFalse(self.cache.exists())

    def test_embedding_diagnostics_are_private_and_saved_text_retries_without_summary(self):
        import autoresearch_vendi.runtime as runtime

        class NotFoundError(RuntimeError):
            status_code = 404

        rows = self.pair()
        for row in rows:
            row.pop("embedding")
            row.pop("embedding_model")
        self.write_input(rows)
        original = self.input.read_bytes()
        stderr = io.StringIO()
        error = runtime.EmbeddingRequestError(
            NotFoundError("PROVIDER-BODY-SENTINEL api_key=SECRET-KEY-SENTINEL"), attempts=1)
        with patch.object(runtime, "Summarizer", side_effect=AssertionError("No summary for saved text")) as summarizer, \
                patch.object(runtime, "embed_solution_samples", side_effect=error), redirect_stderr(stderr):
            self.assertEqual(self.main(), 2)
        summarizer.assert_not_called()
        self.assertEqual(self.input.read_bytes(), original)
        manifest = json.loads((self.output / "manifest.json").read_text())
        diagnostic = manifest["embedding"]["diagnostic"]
        self.assertEqual(manifest["summary_api_calls"], 0)
        for detail in ("NotFoundError", "HTTP 404", "attempts=1"):
            self.assertIn(detail, diagnostic)
            self.assertIn(detail, stderr.getvalue())
        self.assertIn("Embedding error:", stderr.getvalue())
        exported = self.output / "samples.jsonl"
        failed = [json.loads(line) for line in exported.read_text().splitlines()]
        self.assertEqual([row["text"] for row in failed], [row["text"] for row in rows])
        self.assertTrue(all(row["extraction_status"] == "error" and "embedding" not in row for row in failed))
        artifacts = stderr.getvalue() + "".join(path.read_text() for path in self.output.iterdir() if path.is_file())
        for secret in ("PROVIDER-BODY-SENTINEL", "SECRET-KEY-SENTINEL"):
            self.assertNotIn(secret, artifacts)

        self.input, self.output = exported, self.root / "recovered"
        failed_input = self.input.read_bytes()

        def embed(samples, cache, **kwargs):
            self.assertEqual(len(samples), len(rows))
            for row in samples:
                self.assertEqual(row["extraction_status"], "ok")
                self.assertNotIn("error", row)
                self.assertEqual(row["text"], "A source-grounded method description.")
                row.update(embedding=[1, 0], embedding_model="fixture-retried-v1")
            return {"backend": "fixture", "model": "fixture-retried-v1"}

        with patch.object(runtime, "Summarizer", side_effect=AssertionError("No summary on retry")) as summarizer, \
                patch.object(runtime, "embed_solution_samples", side_effect=embed) as embeddings:
            self.assertEqual(self.main(), 0)
        summarizer.assert_not_called()
        embeddings.assert_called_once()
        self.assertEqual(self.input.read_bytes(), failed_input)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["summary_api_calls"], 0)
        self.assertEqual(manifest["embedding"]["backend"], "fixture")

    def test_eleven_against_seven_enumerates_all_330_subsets_at_m_seven(self):
        rows = [self.sample("baseline", str(i), [1, 0, 0, 0, 0, 0, 0]) for i in range(11)]
        rows += [self.sample("analogy", str(i), [int(i == j) for j in range(7)]) for i in range(7)]
        self.write_input(rows)
        self.assertEqual(self.main(), 0)
        scores = self.csv_rows("run_scores.csv")
        matched = {row["arm"]: row for row in scores if row["m"] == "7"}
        self.assertEqual(matched["baseline"]["n_subsets"], "330")
        self.assertEqual(matched["analogy"]["n_subsets"], "1")
        self.assertEqual(matched["baseline"]["n_total"], "11")
        self.assertEqual(matched["analogy"]["n_total"], "7")
        self.assertAlmostEqual(float(matched["baseline"]["vendi"]), 1.0)
        self.assertAlmostEqual(float(matched["analogy"]["vendi"]), 7.0)
        self.assertEqual({row["m"] for row in scores}, {str(n) for n in range(2, 8)})
        full = {row["arm"]: row for row in self.csv_rows("full_run_scores.csv")}
        self.assertEqual((full["baseline"]["n"], full["analogy"]["n"]), ("11", "7"))
        self.assertTrue(all(row["status"] == "descriptive_unmatched_counts" for row in full.values()))

    def test_multiple_tasks_and_pairs_remain_separate(self):
        rows = []
        for task in ("task-a", "task-b"):
            for pair in ("pair-one", "pair-two"):
                for row in self.pair(task=task, pair_id=pair):
                    row["run_id"] = task + "-" + pair + "-" + row["arm"]
                    rows.append(row)
        self.write_input(rows)
        self.assertEqual(self.main(), 0)
        self.assertEqual(len(self.csv_rows("run_scores.csv")), 8)
        self.assertEqual(len({row["cohort"] for row in self.csv_rows("run_scores.csv")}), 2)
        means = [row for row in self.csv_rows("comparisons.csv") if row["kind"] == "paired_mean"]
        self.assertEqual(len(means), 2)
        self.assertTrue(all(row["n_pairs"] == "2" and abs(float(row["delta"]) - 1) < 1e-8 for row in means))

    def test_solution_mode_rejects_parent_and_stage_options(self):
        self.make_run()
        for options in (("--parent-map", str(self.root / "missing.csv")), ("--stages", "improve")):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.main(*options, "--prepare-only", raw=True, stdlib=True)
        self.assertFalse(self.output.exists())

    def test_input_modes_and_existing_output_mode_cannot_be_mixed(self):
        self.write_input(self.pair())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.main("--mode", "changes", "--prepare-only", stdlib=True)
        self.write_input([{**row, "stage": "improve", "representation_version": "diff-v1"} for row in self.pair()])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.main("--prepare-only", stdlib=True)
        self.write_input(self.pair())
        self.output.mkdir()
        manifest = self.output / "manifest.json"
        manifest.write_text(json.dumps({"settings": {"mode": "changes"}}))
        before = self.snapshot(self.output)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.main("--prepare-only", stdlib=True)
        self.assertEqual(self.snapshot(self.output), before)

    def test_output_guards_preserve_raw_and_normalized_inputs(self):
        run = self.make_run()
        before = self.snapshot(self.raw)
        for option in ("--out", "--cache"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.main(option, str(run / "nested"), "--prepare-only", raw=True, stdlib=True)
        self.assertEqual(self.snapshot(self.raw), before)
        self.output.mkdir()
        self.input = self.output / "samples.jsonl"
        self.write_input(self.pair())
        original = self.input.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.main("--prepare-only", stdlib=True)
        self.assertEqual(self.input.read_bytes(), original)

    def test_full_flow_supplies_only_numbered_source_to_new_summarizer(self):
        import autoresearch_vendi.runtime as runtime
        self.make_run()
        self.make_run("analogy")
        before = self.snapshot(self.raw)
        observed = []

        def summarize(instance, source):
            self.assertIn(source, {"SOURCE:1: value = 1", "SOURCE:1: value = 2"})
            observed.append(source)
            instance.calls += 1
            return {"title": "Scalar computation", "summary": "Computes one scalar value.",
                    "evidence": ["SOURCE:1"]}, 1

        def embed(samples, cache, *, model_name, **kwargs):
            self.assertEqual(model_name, "text-embedding-3-small")
            for row in samples:
                self.assertEqual(row["text"], "Computes one scalar value.")
                self.assertEqual(row["stage"], "solution")
                self.assertNotIn("assessment_status", row)
                row.update(embedding=[1, 0] if row["candidate_id"] == "trial0001" else [0, 1],
                           embedding_model="fixture-vectors-v1")
            return {"backend": "fixture", "model": model_name}

        with patch.object(runtime.Summarizer, "summarize_solution", autospec=True, side_effect=summarize), \
                patch.object(runtime.Summarizer, "summarize", side_effect=AssertionError("legacy summary")) as old_summary, \
                patch.object(runtime.Summarizer, "assess_change", side_effect=AssertionError("diff assessment")) as old_diff, \
                patch("autoresearch_vendi.changes.build_change_packet", side_effect=AssertionError("diff packet")) as packet, \
                patch.object(runtime, "embed_solution_samples", side_effect=embed) as embeddings:
            self.assertEqual(self.main(raw=True), 0)
        self.assertEqual(len(observed), 4)
        old_summary.assert_not_called()
        old_diff.assert_not_called()
        packet.assert_not_called()
        embeddings.assert_called_once()
        self.assertEqual(self.snapshot(self.raw), before)
        self.assertFalse((self.output / "change_evidence").exists())
        self.assertFalse((self.output / "parent_map.csv").exists())
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["summary_api_calls"], 4)
        self.assertEqual(len(self.csv_rows("solution_cards.csv")), 4)
        self.assertEqual(len(self.csv_rows("full_run_scores.csv")), 2)


if __name__ == "__main__":
    unittest.main()
