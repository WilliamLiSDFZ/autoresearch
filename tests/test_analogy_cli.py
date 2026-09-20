"""CLI lifecycle and persistence checks using only synthetic files and mocked inference."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoresearch_analogy import cli
from autoresearch_analogy.agent import AnalogyResult
from autoresearch_analogy.context import build_packet


FAKE_KEY = "fixture-api-credential-never-persist"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class AnalogyCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="analogy-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Freeze checks run against isolated fixture sources, not concurrently edited repo files.
        self.code = self.root / "fixed-code"
        self.code.mkdir()
        (self.code / "autoresearch_analogy").mkdir()
        for name in ("analogy_agent.py", "prepare.py", "pyproject.toml", "uv.lock"):
            (self.code / name).write_text("fixture frozen code\n", encoding="utf-8")
        (self.code / "autoresearch_analogy" / "agent.py").write_text("fixture agent\n", encoding="utf-8")
        self.addCleanup(patch.stopall)
        patch.object(cli, "ROOT", self.code).start()
        patch.dict(os.environ, {"ANALOGY_API_KEY": FAKE_KEY, "ANALOGY_MODEL": "fixture-model"}, clear=True).start()
        self.loop = patch("autoresearch_analogy.observed_loop.run",
                          side_effect=AssertionError("Unexpected live inference" )).start()
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.manifest = {"version": 1, "task_id": "jigsaw-unintended-bias-in-toxicity-classification",
            "metric_version": "jubias-continuous-auc-v1", "maximize": True,
            "seed": 42, "validation_fraction": 0.05, "train_rows": 950,
            "validation_rows": 50, "test_rows": 100}
        self.manifest["prepared_id"] = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write_json(self.prepared / "manifest.json", self.manifest)
        self.task = self.root / "description.md"
        self.task.write_text("# Task\nFixed subgroup AUC classification.\n", encoding="utf-8")
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        paper = {"id": "venue/fixture", "title": "Group Risk and Sampling",
            "venue": "venue", "category": "sampling", "categories": ["sampling"],
            "abstract": "Sampling distribution changes the risk of minority groups.",
            "tldr": "Correct sampling to control group risk.",
            "source": "https://arxiv.org/abs/2407.13957v1", "pdf_url": ""}
        raw = (json.dumps(paper) + "\n").encode()
        (self.corpus / "records.jsonl").write_bytes(raw)
        write_json(self.corpus / "manifest.json", {"level": "paper", "schema_version": 2,
            "count": 1, "venues": {"venue": 1}, "records_sha1": hashlib.sha1(raw).hexdigest()[:12]})
        self.source = self.root / "train.py"
        self.source.write_text("raise RuntimeError('must never be executed by retrieval')\n", encoding="utf-8")

    def call(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main([str(value) for value in args])
        self.assertNotIn(FAKE_KEY, stdout.getvalue())
        self.assertNotIn(FAKE_KEY, stderr.getvalue())
        return code, stdout.getvalue(), stderr.getvalue()

    def common(self, command="run", stage="draft", output=None, extra=()):
        args = [command, "--stage", stage, "--task-file", self.task,
                "--prepared-dir", self.prepared, "--corpus-dir", self.corpus,
                "--cache-dir", self.root / "paper-cache", "--offline"]
        if command == "run":
            args += ["--output-dir", output or self.root / "report"]
        return args + list(extra)

    def snapshot(self, name="candidate"):
        directory = self.root / name
        result = self.call(["snapshot", "--source", self.source, "--artifact-dir", directory,
            "--prepared-dir", self.prepared, "--experiment-id", name, "--run-id", "run_001"])
        self.assertEqual(result[0], 0, result)
        return directory

    def candidate_outputs(self, directory, **changes):
        metrics = {"score": 0.8, "overall_auc": 0.9, "power_means": {"subgroup": 0.8},
            "identities": {}, "prepared_id": self.manifest["prepared_id"],
            "metric_version": self.manifest["metric_version"], "maximize": True,
            "validation_rows": 50, **changes}
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "config.json", {"learning_rate": 0.01})

    def complete(self, directory, *extra):
        return self.call(["complete", "--artifact-dir", directory,
                          "--prepared-dir", self.prepared, *extra])

    def test_snapshot_complete_and_improve_bind_exact_source_and_results(self):
        directory = self.snapshot()
        pending = json.loads((directory / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(pending["execution_status"], "pending")
        self.assertEqual((directory / "source.py").read_bytes(), self.source.read_bytes())
        self.candidate_outputs(directory)
        log = self.root / "training.log"
        log.write_text("loss=0.3\ncompleted\n", encoding="utf-8")
        self.assertEqual(self.complete(directory, "--log-file", log)[0], 0)
        receipt = json.loads((directory / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["execution_status"], "completed")
        self.assertEqual(receipt["metrics_sha256"], sha(directory / "metrics.json"))
        self.assertEqual(receipt["config_sha256"], sha(directory / "config.json"))
        self.assertEqual(receipt["log_sha256"], sha(directory / "run.log"))
        _, metadata, runtime, session = build_packet("improve", self.task, self.prepared, directory)
        self.assertEqual(metadata["parent_experiment_id"], "candidate")
        self.assertEqual(runtime["public_validation"]["score"], 0.8)
        self.assertTrue(session.sources["candidate"]["available"])
        self.loop.assert_not_called()
        before = (directory / "source.json").read_bytes()
        self.assertEqual(self.complete(directory)[0], 2)
        self.assertEqual((directory / "source.json").read_bytes(), before)
        self.loop.side_effect = None
        self.loop.return_value = AnalogyResult(delivery_status="abstained", reason="No fitting mechanism")
        result = self.call(self.common(stage="improve", extra=["--parent-artifacts", directory]))
        self.assertEqual(result[0], 0, result)
        self.assertEqual(self.loop.call_args.kwargs["mode"], "improve")
        self.assertEqual(self.loop.call_args.kwargs["packet_metadata"]["parent_experiment_id"], "candidate")
        self.assertEqual(self.loop.call_args.kwargs["runtime_context"]["public_validation"]["score"], 0.8)

    def test_snapshot_will_not_overwrite_existing_directory(self):
        directory = self.snapshot()
        before = (directory / "source.py").read_bytes()
        result = self.call(["snapshot", "--source", self.source, "--artifact-dir", directory,
            "--prepared-dir", self.prepared, "--experiment-id", "candidate", "--run-id", "run_001"])
        self.assertEqual(result[0], 2)
        self.assertEqual((directory / "source.py").read_bytes(), before)

    def test_complete_rejects_changed_source_and_wrong_metrics(self):
        directory = self.snapshot()
        self.candidate_outputs(directory)
        original = self.source.read_text(encoding="utf-8")
        self.source.write_text("edited_after_snapshot = True\n", encoding="utf-8")
        self.assertEqual(self.complete(directory)[0], 2)
        self.source.write_text(original, encoding="utf-8")
        for changes in ({"prepared_id": "wrong"}, {"metric_version": "wrong"},
                        {"score": float("nan")}, {"score": 1.1}, {"maximize": False}):
            with self.subTest(changes=changes):
                self.candidate_outputs(directory, **changes)
                self.assertEqual(self.complete(directory)[0], 2)
                self.assertEqual(json.loads((directory / "source.json").read_text(encoding="utf-8"))["execution_status"], "pending")
        self.loop.assert_not_called()

    def test_preflight_creates_lock_without_api_and_refuses_overwrite(self):
        lock = self.root / "protocol-lock.json"
        args = self.common("preflight", extra=["--lock-file", lock])
        code, stdout, _ = self.call(args)
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(stdout)["model_contacted"])
        self.assertTrue(json.loads(stdout)["fulltext"])
        frozen = lock.read_bytes()
        self.assertNotIn(FAKE_KEY, frozen.decode())
        self.assertEqual(self.call(args)[0], 2)
        self.assertEqual(lock.read_bytes(), frozen)
        self.loop.assert_not_called()

    def test_run_rejects_changed_frozen_code_before_model_call(self):
        lock = self.root / "protocol-lock.json"
        self.assertEqual(self.call(self.common("preflight", extra=["--lock-file", lock]))[0], 0)
        (self.code / "autoresearch_analogy" / "agent.py").write_text("changed frozen code\n", encoding="utf-8")
        output = self.root / "report"
        code, _, stderr = self.call(self.common(output=output, extra=["--lock-file", lock]))
        self.assertEqual(code, 2)
        self.assertIn("Frozen protocol differs", stderr)
        self.assertFalse(output.exists())
        self.loop.assert_not_called()

    def test_run_will_not_overwrite_existing_output(self):
        output = self.root / "report"
        output.mkdir()
        marker = output / "report.md"
        marker.write_text("prior immutable report\n", encoding="utf-8")
        self.assertEqual(self.call(self.common(output=output))[0], 2)
        self.assertEqual(marker.read_text(encoding="utf-8"), "prior immutable report\n")
        self.loop.assert_not_called()

    def test_run_persists_success_abstention_and_failure_with_correct_exit(self):
        for delivery, expected, code in (("accepted_complete", "ok", 0),
                                        ("abstained", "abstained", 0), ("failed", "failed", 1)):
            with self.subTest(delivery=delivery):
                output = self.root / delivery
                self.loop.side_effect = None
                self.loop.return_value = AnalogyResult(delivery_status=delivery, reason="fixture reason",
                    report={"mechanisms": [{"mechanism_id": "m1"}]} if delivery.startswith("accepted") else None,
                    report_md="A complete validated report" if delivery.startswith("accepted") else "",
                    failure_kind="transport" if delivery == "failed" else "", turns=2,
                    in_tokens=20, out_tokens=10, seconds=0.1, trace=["fixture tool trace"],
                    queries=["group risk"], paper_ids=["venue/fixture"])
                result = self.call(self.common(output=output))
                self.assertEqual(result[0], code, result)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], expected)
                self.assertEqual(manifest["delivery_status"], delivery)
                for name in ("context.json", "context_budget.json", "trace.jsonl", "report.json", "report.md", "fulltext.json",
                             "code_reads.json", "model_calls.json", "submission_attempts.json"):
                    self.assertTrue((output / name).is_file(), name)
                context = json.loads((output / "context.json").read_text(encoding="utf-8"))
                self.assertIn("packet", context)
                self.assertIn("runtime_context", context)
                self.assertIn("metadata", context)
                self.assertEqual(self.loop.call_args.kwargs["mode"], "draft")
                self.assertTrue(self.loop.call_args.kwargs["fulltext"].enabled)

    def test_all_persisted_inference_fields_redact_credentials(self):
        self.loop.side_effect = None
        self.loop.return_value = AnalogyResult(delivery_status="failed", reason="error " + FAKE_KEY,
            trace=[FAKE_KEY], report={"mechanisms": [], "note": FAKE_KEY}, report_md=FAKE_KEY,
            queries=["query " + FAKE_KEY], paper_ids=[FAKE_KEY], fulltext={"message": FAKE_KEY},
            model_calls=[{"message": FAKE_KEY}], submission_attempts=[{"message": FAKE_KEY}])
        output = self.root / "redacted-report"
        self.assertEqual(self.call(self.common(output=output))[0], 1)
        for path in output.iterdir():
            if path.is_file():
                self.assertNotIn(FAKE_KEY, path.read_text(encoding="utf-8"), path.name)

    def test_changed_input_during_inference_discards_the_report(self):
        def mutate(*args, **kwargs):
            self.task.write_text("Changed task while retrieval was running\n", encoding="utf-8")
            return AnalogyResult(delivery_status="accepted_complete", report={"mechanisms": [{"mechanism_id": "m1"}]},
                                 report_md="This must be discarded")
        self.loop.side_effect = mutate
        output = self.root / "changed-input"
        self.assertEqual(self.call(self.common(output=output))[0], 1)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["failure_kind"], "frozen_input_changed")
        self.assertNotIn("This must be discarded", (output / "report.md").read_text(encoding="utf-8"))

    def test_unexpected_exception_leaves_redacted_failure_manifest(self):
        self.loop.side_effect = RuntimeError("provider error " + FAKE_KEY)
        output = self.root / "exception"
        code, _, stderr = self.call(self.common(output=output))
        self.assertEqual(code, 2)
        self.assertIn("[REDACTED]", stderr)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn(FAKE_KEY, (output / "manifest.json").read_text(encoding="utf-8"))

    def test_preflight_rejects_invalid_fulltext_numbers_and_string_booleans(self):
        config = self.root / "config.json"
        invalid = ({"max_papers": 0}, {"max_read_calls": -1}, {"read_chars": 1.5},
                   {"total_open_seconds": float("inf")}, {"open_timeout_seconds": "60"},
                   {"enabled": "true"}, {"offline": "false"}, {"max_papers": True})
        for values in invalid:
            with self.subTest(values=values):
                write_json(config, {"fulltext": values})
                code, _, _ = self.call(self.common("preflight", extra=["--config", config]))
                self.assertEqual(code, 2)
        self.loop.assert_not_called()

    def test_chat_alias_records_effective_reasoning_separately(self):
        lock = self.root / "alias-lock.json"
        args = self.common("preflight", extra=["--lock-file", lock, "--model", "openai/gpt-5.6-sol",
            "--api", "chat_completions", "--reasoning-effort", "high"])
        result = self.call(args)
        self.assertEqual(result[0], 0, result)
        model = json.loads(lock.read_text(encoding="utf-8"))["model"]
        self.assertEqual(model["requested_reasoning_effort"], "high")
        self.assertEqual(model["effective_reasoning_effort"], "none")
        self.assertEqual(model["effective_api"], "chat_completions")
        self.loop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
