import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch-autoresearch-run.py"
SPEC = importlib.util.spec_from_file_location("fetch_autoresearch_run", SCRIPT)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class FetchAutoresearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.worktrees = self.root / "worktrees"
        self.worktrees.mkdir()
        self.output = self.root / "downloads"

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def write_json(self, path, value):
        return self.write(path, json.dumps(value))

    def make_run(self, name="fixture-baseline", completed=True):
        worktree = self.worktrees / ("worktree-" + name)
        run = worktree / "results" / name
        run.mkdir(parents=True)
        prepared_id = "a" * 64
        self.write_json(run / "run.json", {"run_id": name, "prepared_id": prepared_id})
        self.write(run / "results.tsv", "commit\tval_score\tmemory_gb\tstatus\tdescription\n")
        self.write(worktree / "train.py", "# current implementation\n")
        self.write(worktree / "prepare.py", "# fixed evaluator\n")
        self.write(worktree / "autoresearch_analogy" / "responses.py", "# fixed transport\n")
        trial = run / "trials" / "trial0001"
        self.write(trial / "source.py", "# candidate implementation\n")
        self.write_json(trial / "metrics.json", {
            "score": 0.75, "maximize": True, "prepared_id": prepared_id,
            "metric_version": "jubias-continuous-auc-v1",
        })
        self.write_json(trial / "config.json", {"seed": 1337})
        self.write(trial / "run.log", "val_score: 0.75\n")
        self.write(trial / "validation_predictions.npy", b"\x93NUMPY-fixture-predictions")
        self.write_json(trial / "adoption.json", {"mechanism": None})
        receipt = {
            "protocol": "autoresearch-experiment-v1", "run_id": name,
            "prepared_id": prepared_id, "experiment_id": "trial0001",
            "source_path": "/remote/only/train.py",
            "execution_status": "completed" if completed else "pending",
            "sha256": self.digest(trial / "source.py"),
        }
        if completed:
            for filename, field in (("metrics.json", "metrics_sha256"),
                                    ("config.json", "config_sha256"), ("run.log", "log_sha256")):
                receipt[field] = self.digest(trial / filename)
        self.write_json(trial / "source.json", receipt)
        return worktree, run

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def pack(self, run, full=False):
        output = io.BytesIO()
        fetch.pack_run(self.worktrees, run.relative_to(self.worktrees).as_posix(), full, output)
        return output.getvalue()

    def extract(self, content, name="extracted"):
        archive = self.root / (name + ".tgz")
        archive.write_bytes(content)
        destination = self.root / name
        manifest = fetch.extract_bundle(archive, destination)
        return destination, manifest

    def make_archive(self, entries):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, content, kind in entries:
                info = tarfile.TarInfo(name)
                if kind == "file":
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                else:
                    info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                    info.linkname = content
                    archive.addfile(info)
        return output.getvalue()

    def test_compact_keeps_receipts_predictions_reports_and_source_snapshot(self):
        worktree, run = self.make_run("fixture-analogy")
        for name in ("report.json", "context.json", "manifest.json", "model_calls.json",
                     "fulltext.json", "code_reads.json", "submission_attempts.json"):
            self.write_json(run / "analogy" / "draft-0001" / name, {"status": "failed"})
        self.write(run / "analogy" / "draft-0001" / "trace.jsonl", "{}\n")
        self.write(run / "analogy" / "draft-0001" / "report.md", "No result\n")
        self.write(run / "analogy" / "draft-0001.log", "Transport failed\n")
        self.write_json(run / "protocol.json", {"code_sha256": {}})
        self.write_json(run / "maintenance-001.json", {"reason": "transport fix"})
        self.write(run / "trials" / "trial0001" / "model.safetensors", b"large-model")
        destination, manifest = self.extract(self.pack(run))
        expected = {
            "trials/trial0001/source.py", "trials/trial0001/source.json",
            "trials/trial0001/metrics.json", "trials/trial0001/config.json",
            "trials/trial0001/run.log", "trials/trial0001/adoption.json",
            "trials/trial0001/validation_predictions.npy", "analogy/draft-0001.log",
            "analogy/draft-0001/model_calls.json", "analogy/draft-0001/trace.jsonl",
            "analogy/draft-0001/report.md", "protocol.json", "maintenance-001.json",
            "_worktree/train.py", "_worktree/prepare.py", "_worktree/autoresearch_analogy/responses.py",
        }
        self.assertTrue(expected.issubset(manifest["files"]))
        self.assertFalse((destination / "trials/trial0001/model.safetensors").exists())
        self.assertEqual((destination / "_worktree/train.py").read_bytes(), (worktree / "train.py").read_bytes())
        self.assertEqual(fetch.verify_receipts(destination), [])

    def test_full_adds_weights_but_never_caches_environments_or_datasets(self):
        _, run = self.make_run()
        self.write(run / "trials/trial0001/model.pt", b"model")
        self.write(run / "trials/trial0001/optimizer.bin", b"optimizer")
        excluded = ("analogy-cache/paper/paper.pdf", ".venv/lib/package.py",
                    "jigsaw-data/seed-42/train.csv", "datasets/train.csv",
                    "paper-corpus/records.jsonl", ".git/config", "trials/trial0001/cache/tokenized.npy")
        for name in excluded:
            self.write(run / name, b"must stay remote")
        for full in (False, True):
            with self.subTest(full=full):
                destination, manifest = self.extract(self.pack(run, full), "full-" + str(full))
                self.assertEqual((destination / "trials/trial0001/model.pt").exists(), full)
                self.assertEqual((destination / "trials/trial0001/optimizer.bin").exists(), full)
                self.assertEqual(manifest["full"], full)
                for name in excluded:
                    self.assertFalse((destination / name).exists(), name)

    def test_symlinks_outside_run_are_not_downloaded(self):
        worktree, run = self.make_run()
        secret = self.write(self.root / "outside/credential.txt", "private fixture")
        (run / "outside-file.txt").symlink_to(secret)
        (run / "outside-directory").symlink_to(secret.parent, target_is_directory=True)
        (worktree / "program.md").symlink_to(secret)
        destination, manifest = self.extract(self.pack(run, full=True))
        self.assertFalse((destination / "outside-file.txt").exists())
        self.assertFalse((destination / "outside-directory").exists())
        self.assertFalse((destination / "_worktree/program.md").exists())
        self.assertIn("outside-file.txt", manifest["excluded"])
        self.assertIn("outside-directory/", manifest["excluded"])
        self.assertEqual(secret.read_text(), "private fixture")

    def test_discovery_treats_pattern_as_literal_and_ignores_symlinked_runs(self):
        literal = "run-$(echo-PWN);*"
        _, selected = self.make_run(literal)
        other_worktree, other = self.make_run("ordinary")
        (other_worktree / "results" / "linked").symlink_to(selected, target_is_directory=True)
        rows = fetch.discover_runs(self.worktrees, "$(echo-PWN);*")
        self.assertEqual([row["name"] for row in rows], [literal])
        self.assertEqual(fetch.discover_runs(self.worktrees, "ordinary*"), [])
        self.assertEqual({row["name"] for row in fetch.discover_runs(self.worktrees)}, {literal, other.name})

    def test_discovery_does_not_traverse_symlinked_trials_or_summary(self):
        _, outside_run = self.make_run("outside")
        worktree = self.worktrees / "linked-trials"
        run = worktree / "results" / "linked-trials-run"
        self.write_json(run / "run.json", {"run_id": run.name})
        (run / "trials").symlink_to(outside_run / "trials", target_is_directory=True)
        (run / "summary.md").symlink_to(outside_run / "run.json")
        row = fetch.discover_runs(self.worktrees, "linked-trials-run")[0]
        self.assertEqual(row["trials"], 0)
        self.assertEqual(row["completed_receipts"], 0)
        self.assertFalse(row["has_summary"])

    def test_pack_rejects_traversal_and_symlinked_run_paths(self):
        worktree, run = self.make_run()
        for relative in ("../results/fixture-baseline", "/tmp/results/fixture-baseline",
                         "worktree/other/fixture-baseline"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                fetch.pack_run(self.worktrees, relative, False, io.BytesIO())
        linked = worktree / "results" / "linked"
        linked.symlink_to(run, target_is_directory=True)
        with self.assertRaises(ValueError):
            fetch.pack_run(self.worktrees, linked.relative_to(self.worktrees).as_posix(), False, io.BytesIO())

    def test_extract_rejects_traversal_absolute_paths_and_links(self):
        cases = (("../escaped.txt", b"escape", "file"),
                 ("/absolute.txt", b"escape", "file"),
                 ("linked", "../escape", "symlink"),
                 ("hardlinked", "../escape", "hardlink"))
        for index, entry in enumerate(cases):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.extract(self.make_archive([entry]), "unsafe-" + str(index))
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_extract_rejects_duplicate_member_and_transfer_hash_corruption(self):
        duplicated = self.make_archive([("same.txt", b"a", "file"), ("same.txt", b"b", "file")])
        with self.assertRaises(ValueError):
            self.extract(duplicated, "duplicate")
        manifest = {"files": {"artifact.txt": hashlib.sha256(b"original").hexdigest()}}
        corrupt = self.make_archive([
            ("artifact.txt", b"modified", "file"),
            ("_fetch.json", json.dumps(manifest).encode(), "file"),
        ])
        with self.assertRaisesRegex(ValueError, "Transfer hash mismatch"):
            self.extract(corrupt, "corrupt")

    def test_extract_never_overwrites_existing_destination(self):
        _, run = self.make_run()
        destination = self.root / "existing"
        sentinel = self.write(destination / "sentinel", b"keep me")
        with self.assertRaises(FileExistsError):
            self.extract(self.pack(run), "existing")
        self.assertEqual(sentinel.read_bytes(), b"keep me")

    def test_receipt_verification_detects_tampered_artifact_and_keeps_receipt(self):
        _, run = self.make_run()
        destination, _ = self.extract(self.pack(run))
        receipt = destination / "trials/trial0001/source.json"
        original_receipt = receipt.read_bytes()
        self.write(destination / "trials/trial0001/source.py", "# changed after completion\n")
        problems = fetch.verify_receipts(destination)
        self.assertTrue(any("source.py: receipt hash mismatch" in problem for problem in problems), problems)
        self.assertEqual(receipt.read_bytes(), original_receipt)

    def test_receipt_verification_preserves_incomplete_trials(self):
        _, run = self.make_run(completed=False)
        trial = run / "trials/trial0001"
        (trial / "metrics.json").unlink()
        (trial / "validation_predictions.npy").unlink()
        destination, _ = self.extract(self.pack(run))
        self.assertEqual(fetch.verify_receipts(destination), [])
        self.assertEqual(json.loads((destination / "trials/trial0001/source.json").read_text())["execution_status"], "pending")

    def test_receipt_verification_rejects_wrong_run_or_metric_identity(self):
        _, run = self.make_run()
        cases = (("source.json", "run_id", "another-run", "protocol/run_id mismatch"),
                 ("source.json", "protocol", "another-protocol", "protocol/run_id mismatch"),
                 ("metrics.json", "prepared_id", "another-dataset", "prepared_id mismatch"),
                 ("metrics.json", "metric_version", "another-metric", "metric protocol mismatch"),
                 ("metrics.json", "maximize", False, "metric protocol mismatch"))
        for index, (filename, field, value, expected) in enumerate(cases):
            with self.subTest(field=field):
                destination, _ = self.extract(self.pack(run), "identity-" + str(index))
                trial = destination / "trials/trial0001"
                path = trial / filename
                payload = json.loads(path.read_text())
                payload[field] = value
                self.write_json(path, payload)
                if filename == "metrics.json":
                    receipt = json.loads((trial / "source.json").read_text())
                    receipt["metrics_sha256"] = self.digest(path)
                    self.write_json(trial / "source.json", receipt)
                problems = fetch.verify_receipts(destination)
                self.assertTrue(any(expected in problem for problem in problems), problems)
                self.assertFalse(any("hash mismatch" in problem for problem in problems), problems)

    def test_main_passes_literal_pattern_as_argv_without_shell(self):
        pattern = "$(touch not-a-command);*"
        rows = [{"name": "fixture", "completed_receipts": 0, "trials": 0, "has_summary": False}]
        with patch.dict(os.environ, {"NS": "fixture-namespace", "CONTEXT": "fixture-context"}), \
                patch.object(fetch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(rows).encode(), b"")) as invoke, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fetch.main(["--list", pattern]), 0)
        call = invoke.call_args
        self.assertEqual(call.args[0][-1], pattern)
        self.assertNotIn("shell", call.kwargs)
        self.assertEqual(call.kwargs["input"], SCRIPT.read_bytes())

    def test_main_existing_snapshot_is_not_downloaded_or_overwritten(self):
        _, run = self.make_run()
        rows = fetch.discover_runs(self.worktrees)
        sentinel = self.write(self.output / run.name / "sentinel", "preserve")
        with patch.object(fetch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(rows).encode(), b"")) as invoke, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fetch.main(["--all", "--out-dir", str(self.output)]), 0)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_batch_pack_failure_does_not_prevent_later_success(self):
        _, failed = self.make_run("failed")
        _, successful = self.make_run("successful")
        listed = {row["name"]: row for row in fetch.discover_runs(self.worktrees)}
        rows = [listed[failed.name], listed[successful.name]]
        archive = self.pack(successful)
        attempted = []

        def fake_run(command, **kwargs):
            if command[-3] == "list":
                return subprocess.CompletedProcess(command, 0, json.dumps(rows).encode(), b"")
            relative = command[-2]
            attempted.append(relative)
            if relative == listed[failed.name]["relative_path"]:
                raise subprocess.CalledProcessError(1, command, stderr=b"fixture pack failure")
            kwargs["stdout"].write(archive)
            return subprocess.CompletedProcess(command, 0, stderr=b"")

        stderr = io.StringIO()
        with patch.object(fetch.subprocess, "run", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = fetch.main(["--all", "--out-dir", str(self.output)])
        self.assertEqual(code, 1)
        self.assertEqual(attempted, [row["relative_path"] for row in rows])
        self.assertFalse((self.output / failed.name).exists())
        self.assertTrue((self.output / successful.name / "results.tsv").is_file())
        self.assertIn("fixture pack failure", stderr.getvalue())
        self.assertEqual(list(self.output.glob(".fetch-*")), [])


if __name__ == "__main__":
    unittest.main()
