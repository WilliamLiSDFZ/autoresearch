"""Offline checks for neutral snapshots and result binding."""

import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import experiment_artifacts as artifacts


def _write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ExperimentArtifactsTests(unittest.TestCase):
    # 创建只有本地合成源码和准备元信息的测试环境。
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="experiment-artifacts-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.manifest = {"version": 1,
            "task_id": "jigsaw-unintended-bias-in-toxicity-classification",
            "metric_version": artifacts.METRIC_VERSION, "maximize": True,
            "seed": 42, "validation_fraction": 0.05,
            "train_rows": 950, "validation_rows": 50, "test_rows": 100}
        self.prepared_id = hashlib.sha256(json.dumps(
            self.manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.manifest["prepared_id"] = self.prepared_id
        _write_json(self.prepared / "manifest.json", self.manifest)
        self.source = self.root / "train.py"
        self.source.write_text("raise RuntimeError('候选代码不得在归档时执行')\n", encoding="utf-8")
        self.run_id = "fixture-run"
        self.output = self.root / "candidate-001"

    def _call(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = artifacts.main([str(value) for value in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def _snapshot_args(self):
        return ["snapshot", "--source", self.source, "--artifact-dir", self.output,
            "--experiment-id", "candidate-001", "--run-id", self.run_id,
            "--prepared-id", self.prepared_id]

    def _snapshot(self):
        result = self._call(*self._snapshot_args())
        self.assertEqual(result[0], 0, result)
        return _read_json(self.output / "source.json")

    def _outputs(self, **changes):
        metrics = {"score": 0.8, "overall_auc": 0.9,
            "metric_version": artifacts.METRIC_VERSION, "maximize": True,
            "prepared_id": self.prepared_id, **changes}
        _write_json(self.output / "metrics.json", metrics)
        _write_json(self.output / "config.json", {"model": "合成测试", "seed": 42})

    def _complete(self, *extra):
        return self._call("complete", "--artifact-dir", self.output, *extra)

    def _assert_pending(self):
        receipt = _read_json(self.output / "source.json")
        self.assertEqual(receipt["execution_status"], "pending")
        self.assertNotIn("completed_at_utc", receipt)
        self.assertNotIn("metrics_sha256", receipt)

    # 验证中性收据的源码、结果和日志绑定可被现有 improve 上下文读取。
    def test_completed_receipt_is_compatible_with_existing_context(self):
        receipt = self._snapshot()
        self.assertEqual(receipt["execution_status"], "pending")
        created_at = datetime.fromisoformat(receipt["created_at_utc"])
        self.assertEqual(created_at.tzinfo, timezone.utc)
        self.assertEqual(receipt["source_path"], str(self.source.resolve()))
        self.assertEqual((self.output / "source.py").read_bytes(), self.source.read_bytes())
        self._outputs()
        log = self.root / "training.log"
        log.write_text("val_score: 0.8\n测试日志\n", encoding="utf-8")
        result = self._complete("--log-file", log, "--prepared-id", self.prepared_id,
                                "--run-id", self.run_id)
        self.assertEqual(result[0], 0, result)
        receipt = _read_json(self.output / "source.json")
        self.assertEqual(receipt["execution_status"], "completed")
        completed_at = datetime.fromisoformat(receipt["completed_at_utc"])
        self.assertEqual(completed_at.tzinfo, timezone.utc)
        self.assertLessEqual(created_at, completed_at)
        for name, field in (("source.py", "sha256"), ("metrics.json", "metrics_sha256"),
                            ("config.json", "config_sha256"), ("run.log", "log_sha256")):
            self.assertEqual(receipt[field], hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        task = self.root / "task.md"
        task.write_text("Classify Jigsaw comments using the fixed evaluator.\n", encoding="utf-8")
        from autoresearch_analogy.context import build_packet
        packet, metadata, runtime, code_session = build_packet(
            "improve", task, self.prepared, self.output)
        self.assertEqual(metadata["parent_experiment_id"], "candidate-001")
        self.assertEqual(metadata["run_id"], self.run_id)
        self.assertEqual(runtime["public_validation"]["score"], 0.8)
        self.assertIsNotNone(code_session)
        self.assertIn("candidate-001", packet)

    # 验证已归档实验不能重复完成。
    def test_duplicate_completion_is_rejected(self):
        self._snapshot()
        self._outputs()
        self.assertEqual(self._complete()[0], 0)
        receipt = (self.output / "source.json").read_bytes()
        self.assertEqual(self._complete()[0], 2)
        self.assertEqual((self.output / "source.json").read_bytes(), receipt)

    # 验证现有目录不会被覆盖，缺少预先快照也不能归档。
    def test_snapshot_does_not_overwrite_and_complete_requires_snapshot(self):
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep", encoding="utf-8")
        result = self._call(*self._snapshot_args())
        self.assertEqual(result[0], 2)
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(self._complete()[0], 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    # 验证原始源码或保存的源码被修改时都不能绑定结果。
    def test_changed_original_or_snapshot_source_is_rejected(self):
        self._snapshot()
        self._outputs()
        for path in (self.source, self.output / "source.py"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_text("changed = True\n", encoding="utf-8")
                result = self._complete()
                self.assertEqual(result[0], 2, result)
                self.assertIn("Source changed", result[2])
                self._assert_pending()
                path.write_bytes(original)

    # 验证无效数值、指标版本、数据身份和最大化方向都不会产生完成收据。
    def test_invalid_metrics_are_rejected(self):
        self._snapshot()
        changes = [{"score": value} for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.8", 10 ** 1000)]
        changes += [{"prepared_id": "0" * 64}, {"metric_version": "other"}, {"maximize": False}]
        for change in changes:
            with self.subTest(change=change):
                self._outputs(**change)
                result = self._complete()
                self.assertEqual(result[0], 2, result)
                self._assert_pending()

    # 验证配置必须是有限 JSON 对象，不能缺失或包含非有限嵌套值。
    def test_invalid_config_or_missing_metrics_is_rejected(self):
        self._snapshot()
        self.assertEqual(self._complete()[0], 2)
        self._assert_pending()
        for config in ([], {"bad": float("nan")}, {"bad": float("inf")}):
            with self.subTest(config=config):
                self._outputs()
                _write_json(self.output / "config.json", config)
                self.assertEqual(self._complete()[0], 2)
                self._assert_pending()
        (self.output / "config.json").write_text('{"overflow": 1e10000}', encoding="utf-8")
        self.assertEqual(self._complete()[0], 2)

    # 验证快照身份合法，并核对完成命令中显式提供的运行和数据身份。
    def test_run_and_prepared_identity_binding(self):
        for flag, value in (("--run-id", " "), ("--prepared-id", "bad"),
                            ("--experiment-id", "../invalid")):
            with self.subTest(flag=flag):
                args = self._snapshot_args()
                args[args.index(flag) + 1] = value
                self.assertEqual(self._call(*args)[0], 2)
                self.assertFalse(self.output.exists())
        self._snapshot()
        self._outputs()
        self.assertEqual(self._complete("--run-id", "another-run")[0], 2)
        self.assertEqual(self._complete("--prepared-id", "0" * 64)[0], 2)
        self._assert_pending()
        self.assertEqual(self._complete()[0], 0)

    # 验证已存在的日志不会被另一份日志覆盖。
    def test_conflicting_log_is_rejected_and_existing_log_is_bound(self):
        self._snapshot()
        self._outputs()
        (self.output / "run.log").write_text("original\n", encoding="utf-8")
        other = self.root / "other.log"
        other.write_text("different\n", encoding="utf-8")
        self.assertEqual(self._complete("--log-file", other)[0], 2)
        self._assert_pending()
        self.assertEqual((self.output / "run.log").read_text(encoding="utf-8"), "original\n")
        self.assertEqual(self._complete()[0], 0)
        self.assertIn("log_sha256", _read_json(self.output / "source.json"))

    # 验证源码和结果文件不能通过符号链接绕过不可变快照。
    def test_symlink_artifact_is_rejected(self):
        self._snapshot()
        self._outputs()
        config = self.output / "config.json"
        outside = self.root / "outside.json"
        outside.write_bytes(config.read_bytes())
        config.unlink()
        config.symlink_to(outside)
        self.assertEqual(self._complete()[0], 2)
        self._assert_pending()

    # 验证工具只导入标准库，并可在禁用站点包的解释器中启动。
    def test_cli_uses_only_standard_library(self):
        source = Path(artifacts.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split(".")[0])
        self.assertTrue(set(imports).issubset(sys.stdlib_module_names), imports)
        result = subprocess.run([sys.executable, "-S", str(source), "--help"],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("snapshot", result.stdout)


if __name__ == "__main__":
    unittest.main()
