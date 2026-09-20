"""Neutral, standard-library bookkeeping for Jigsaw experiments.

This tool never imports candidate code, launches training, or accesses retrieval.
The caller writes metrics.json/config.json into the new snapshot directory, then
finalizes it before editing the source. Kubernetes controls the Job lifetime;
creation and completion timestamps are recorded only for result review.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


METRIC_VERSION = "jubias-continuous-auc-v1"
PROTOCOL = "autoresearch-experiment-v1"


def _bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path.name}")
    return path.read_bytes()


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _json_object(content, name):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value in {name}")

    def number(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}")
            result[key] = value
        return result

    result = json.loads(content.decode("utf-8"), parse_constant=invalid,
                        parse_float=number, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {name}")
    return result


def _read_json(path):
    return _json_object(_bytes(path), Path(path).name)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _identity(args, receipt=None):
    run_id = args.run_id if receipt is None else receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be nonempty")
    if receipt is not None and args.run_id is not None and args.run_id != run_id:
        raise ValueError("run_id differs from the snapshot")
    prepared_id = args.prepared_id if receipt is None else receipt.get("prepared_id")
    if not isinstance(prepared_id, str) or not re.fullmatch(r"[0-9a-f]{64}", prepared_id):
        raise ValueError("prepared_id must be the fixed manifest's SHA-256 identity")
    if receipt is not None and args.prepared_id is not None and args.prepared_id != prepared_id:
        raise ValueError("prepared_id differs from the snapshot")
    return prepared_id


# 保存训练前源码，并创建不可覆盖的实验目录。
def snapshot_command(args):
    prepared_id = _identity(args)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_id):
        raise ValueError("experiment_id must contain only letters, digits, underscores, or hyphens")
    original = Path(args.source).expanduser()
    content = _bytes(original)
    content.decode("utf-8")
    source = original.resolve()
    directory = Path(args.artifact_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "source.py").open("xb") as stream:
        stream.write(content)
    receipt = {"protocol": PROTOCOL, "sha256": _hash(content), "source_path": str(source),
               "experiment_id": args.experiment_id, "run_id": args.run_id,
               "prepared_id": prepared_id, "execution_status": "pending",
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(directory / "source.json", receipt)
    print(json.dumps({"status": "pending", "artifact_dir": str(directory),
                      "source_sha256": receipt["sha256"]}))
    return 0


# 验证源码和结果，将指标、配置及可选日志绑定到源码快照。
def complete_command(args):
    directory = Path(args.artifact_dir).expanduser().resolve()
    receipt = _read_json(directory / "source.json")
    if receipt.get("protocol") != PROTOCOL or receipt.get("execution_status") != "pending":
        raise ValueError("Only a pending neutral snapshot may be completed")
    prepared_id = _identity(args, receipt)
    source_hash = receipt.get("sha256")
    if not isinstance(receipt.get("source_path"), str) or not receipt["source_path"]:
        raise ValueError("Snapshot is missing its original source path")
    if (_hash(_bytes(directory / "source.py")) != source_hash
            or _hash(_bytes(receipt["source_path"])) != source_hash):
        raise ValueError("Source changed after snapshot; results cannot be bound")
    artifacts = {name: _bytes(directory / name) for name in ("metrics.json", "config.json")}
    metrics = _json_object(artifacts["metrics.json"], "metrics.json")
    _json_object(artifacts["config.json"], "config.json")
    score = metrics.get("score")
    if not _finite_number(score) or not 0 <= score <= 1:
        raise ValueError("Metrics require a finite score in [0, 1]")
    if metrics.get("maximize") is not True or metrics.get("metric_version") != METRIC_VERSION:
        raise ValueError("Metrics must maximize the fixed Jigsaw metric_version")
    if metrics.get("prepared_id") != prepared_id:
        raise ValueError("Metrics prepared_id differs from the snapshot")
    log = directory / "run.log"
    if args.log_file:
        content = _bytes(Path(args.log_file).expanduser())
        if log.exists() or log.is_symlink():
            if _bytes(log) != content:
                raise ValueError("A different run.log already exists in this experiment directory")
        else:
            with log.open("xb") as stream:
                stream.write(content)
    if log.exists() or log.is_symlink():
        artifacts["run.log"] = _bytes(log)
    receipt.update(metrics_sha256=_hash(artifacts["metrics.json"]),
                   config_sha256=_hash(artifacts["config.json"]))
    if "run.log" in artifacts:
        receipt["log_sha256"] = _hash(artifacts["run.log"])
    receipt.update(execution_status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    temporary = directory / "source.json.tmp"
    _write_json(temporary, receipt)
    try:
        temporary.replace(directory / "source.json")
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"status": "completed", "artifact_dir": str(directory), "score": score}))
    return 0


# 定义中性快照和结果归档命令的参数。
def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Save source before a candidate executes")
    snapshot.add_argument("--source", type=Path, required=True)
    snapshot.add_argument("--experiment-id", required=True)
    complete = commands.add_parser("complete", help="Bind completed metrics to the saved source")
    complete.add_argument("--log-file", type=Path)
    for command in (snapshot, complete):
        command.add_argument("--artifact-dir", type=Path, required=True)
        command.add_argument("--run-id", required=command is snapshot)
        command.add_argument("--prepared-id", required=command is snapshot)
    snapshot.set_defaults(handler=snapshot_command)
    complete.set_defaults(handler=complete_command)
    return result


# 分发命令并将校验失败转换为非零退出码。
def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
