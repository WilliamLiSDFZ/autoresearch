#!/usr/bin/env python3
"""Fetch autoresearch worktree results through the Nautilus dev Pod (stdlib only)."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".cache", "cache", "caches",
                 "analogy-cache", "jigsaw-data", "datasets", "paper-corpus", "paper_corpus"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".pkl", ".pickle",
                   ".joblib", ".h5", ".hdf5", ".onnx"}
SOURCE_FILES = ("train.py", "prepare.py", "experiment_artifacts.py", "analogy_agent.py",
                "program.md", "program-analogy.md", "pyproject.toml", "uv.lock")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def discover_runs(root, pattern=""):
    """Find actual result directories, including unsuccessful setup attempts."""
    runs = []
    for worktree in root.iterdir():
        results = worktree / "results"
        if worktree.is_symlink() or results.is_symlink() or not results.is_dir():
            continue
        for run in results.iterdir():
            if run.is_symlink() or not run.is_dir() or pattern not in run.name:
                continue
            markers = [run / name for name in ("run.json", "results.tsv", "summary.md")]
            markers = [p for p in markers if p.is_file() and not p.is_symlink()]
            if not markers:
                continue
            completed = 0
            trial_root = run / "trials"
            trials = ([] if trial_root.is_symlink() else
                      [p for p in trial_root.glob("*") if p.is_dir() and not p.is_symlink()])
            for trial in trials:
                receipt = trial / "source.json"
                if receipt.is_file() and not receipt.is_symlink():
                    try:
                        completed += _json(receipt).get("execution_status") == "completed"
                    except (ValueError, AttributeError):
                        pass  # Keep malformed/incomplete artifacts available for inspection.
            runs.append({"name": run.name, "relative_path": run.relative_to(root).as_posix(),
                         "mtime": max(p.stat().st_mtime for p in markers),
                         "trials": len(trials), "completed_receipts": completed,
                         "has_summary": run / "summary.md" in markers})
    return sorted(runs, key=lambda item: (item["mtime"], item["relative_path"]), reverse=True)


def _run_path(root, relative):
    parts = PurePosixPath(relative).parts
    if len(parts) != 3 or parts[1] != "results" or any(p in (".", "..", "/") for p in parts):
        raise ValueError("Expected worktree/results/run under WORKTREES_DIR")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Run path must not contain symlinks")
    if not current.is_dir():
        raise ValueError("Run directory is missing")
    return current


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def pack_run(root, relative, full, output):
    """Stream one archive without creating or modifying any remote files."""
    run = _run_path(root, relative)
    worktree = run.parent.parent
    manifest = {"format": 1, "run_dir": str(run), "worktree": str(worktree),
                "full": full, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": {}, "excluded": []}
    selected = []
    for directory, dirs, files in os.walk(run, followlinks=False):
        directory = Path(directory)
        for name in list(dirs):
            path = directory / name
            if name in EXCLUDED_DIRS or path.is_symlink():
                dirs.remove(name)
                manifest["excluded"].append(path.relative_to(run).as_posix() + "/")
        for name in sorted(files):
            path = directory / name
            relative_name = path.relative_to(run).as_posix()
            if path.is_symlink() or not path.is_file() or (not full and path.suffix.lower() in WEIGHT_SUFFIXES):
                manifest["excluded"].append(relative_name)
                continue
            if relative_name == "_fetch.json" or relative_name.startswith("_worktree/"):
                raise ValueError("Run uses a reserved download metadata path")
            selected.append((path, relative_name))
    sources = [worktree / name for name in SOURCE_FILES]
    module_dir = worktree / "autoresearch_analogy"
    if not module_dir.is_symlink():
        sources.extend(sorted(module_dir.glob("*.py")))
    selected.extend((path, "_worktree/" + path.relative_to(worktree).as_posix())
                    for path in sources if path.is_file() and not path.is_symlink())
    with tarfile.open(fileobj=output, mode="w|gz") as archive:
        for path, name in selected:
            before = path.stat()
            info = archive.gettarinfo(str(path), arcname=name)
            if not info.isfile():
                raise ValueError("Artifact changed type while packing: " + name)
            with path.open("rb") as stream:
                reader = _HashingReader(stream)
                archive.addfile(info, reader)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Artifact changed while packing: " + name)
            manifest["files"][name] = reader.digest.hexdigest()
        content = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("_fetch.json")
        info.size, info.mode = len(content), 0o644
        archive.addfile(info, io.BytesIO(content))


def extract_bundle(archive_path, destination):
    """Extract regular files only into a fresh staging directory and verify bytes."""
    destination.mkdir()
    names = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
                raise ValueError("Unsafe archive path")
            if not member.isfile() or member.name in names:
                raise ValueError("Archive contains a link, special file or duplicate")
            names.add(member.name)
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)
    manifest = _json(destination / "_fetch.json")
    if names != set(manifest["files"]) | {"_fetch.json"}:
        raise ValueError("Archive file list differs from transfer manifest")
    for name, expected in manifest["files"].items():
        if _digest(destination / name) != expected:
            raise ValueError("Transfer hash mismatch: " + name)
    return manifest


def verify_receipts(directory):
    """Check completed source/result bindings without changing original receipts."""
    problems = []
    run_id = PurePosixPath(_json(directory / "_fetch.json")["run_dir"]).name
    for path in sorted((directory / "trials").glob("*/source.json")):
        try:
            receipt = _json(path)
            if receipt.get("execution_status") != "completed":
                continue
            if receipt.get("protocol") != "autoresearch-experiment-v1" or receipt.get("run_id") != run_id:
                problems.append(path.parent.name + ": receipt protocol/run_id mismatch")
            bindings = {"source.py": "sha256", "metrics.json": "metrics_sha256",
                        "config.json": "config_sha256"}
            if "log_sha256" in receipt:
                bindings["run.log"] = "log_sha256"
            for filename, field in bindings.items():
                if _digest(path.parent / filename) != receipt.get(field):
                    problems.append(path.parent.name + "/" + filename + ": receipt hash mismatch")
            metrics = _json(path.parent / "metrics.json")
            if metrics.get("prepared_id") != receipt.get("prepared_id"):
                problems.append(path.parent.name + ": prepared_id mismatch")
            if metrics.get("metric_version") != "jubias-continuous-auc-v1" or metrics.get("maximize") is not True:
                problems.append(path.parent.name + ": metric protocol mismatch")
            if not (path.parent / "validation_predictions.npy").is_file():
                problems.append(path.parent.name + ": validation_predictions.npy missing")
        except (OSError, ValueError, AttributeError) as exc:
            problems.append(path.parent.name + ": " + str(exc))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Environment overrides: POD, NS, CONTEXT, WORKTREES_DIR, OUT_DIR, REMOTE_PYTHON. "
        "Existing local snapshots are never overwritten; use another OUT_DIR to fetch again."))
    parser.add_argument("pattern", nargs="?", default="", help="literal substring of the run name")
    parser.add_argument("--list", action="store_true", help="list matching runs without downloading")
    parser.add_argument("--all", action="store_true", help="fetch every match (default: newest match)")
    parser.add_argument("--full", action="store_true", help="also include model weights/checkpoints")
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get(
        "OUT_DIR", str(Path.home() / "nautilus" / "autoresearch-result"))))
    args = parser.parse_args(argv)
    pod = os.environ.get("POD", "mlevolve-agentic-knowledge-base-dev-cpu")
    root = os.environ.get("WORKTREES_DIR", "/workspace/autoresearch/worktrees")
    command = ["kubectl", "--context", os.environ.get("CONTEXT", "nautilus"),
               "-n", os.environ.get("NS", "ecepxie"), "exec", "-i", pod, "--",
               os.environ.get("REMOTE_PYTHON", "python"), "-B", "-", "--remote"]
    script = Path(__file__).read_bytes()
    listed = subprocess.run(command + ["list", root, args.pattern], input=script,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    runs = json.loads(listed.stdout)
    if not runs:
        print("No matching runs in " + root, file=sys.stderr)
        return 1
    if args.list:
        for run in runs:
            print("{}  completed_receipts={}/{}  summary={}".format(
                run["name"], run["completed_receipts"], run["trials"], run["has_summary"]))
        return 0
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for run in runs if args.all else runs[:1]:
        name = run["name"]
        if name in (".", "..") or "/" in name or not name:
            raise ValueError("Invalid run name returned by Pod")
        destination = args.out_dir / name
        if destination.exists() or destination.is_symlink():
            print("SKIP existing snapshot: " + str(destination))
            continue
        print("Fetching " + name, flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".fetch-", dir=args.out_dir) as staging:
                staging = Path(staging)
                archive = staging / "run.tgz"
                with archive.open("wb") as output:
                    subprocess.run(command + ["pack", root, run["relative_path"], str(int(args.full))],
                                   input=script, stdout=output, stderr=subprocess.PIPE, check=True)
                manifest = extract_bundle(archive, staging / "result")
                problems = verify_receipts(staging / "result")
                manifest.update(pod=pod, context=command[2], namespace=command[4],
                                archive_sha256=_digest(archive), receipt_errors=problems)
                (staging / "result" / "_fetch.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError("Destination appeared during download")
                (staging / "result").rename(destination)
                print("  Saved {} files ({:.1f} MiB compressed) -> {}".format(
                    len(manifest["files"]), archive.stat().st_size / 1024**2, destination))
                if problems:
                    failures.append(name)
                    print("  Receipt errors (artifacts preserved): " + "; ".join(problems), file=sys.stderr)
        except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as exc:
            failures.append(name)
            detail = exc.stderr.decode(errors="replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            print("  FAILED " + name + ": " + detail, file=sys.stderr)
    if failures:
        print("Runs needing attention: " + ", ".join(failures), file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--remote":
            action, root, selected = sys.argv[2:5]
            if action == "list":
                print(json.dumps(discover_runs(Path(root), selected)))
            elif action == "pack":
                pack_run(Path(root), selected, sys.argv[5] == "1", sys.stdout.buffer)
            else:
                raise ValueError("Unknown remote action")
        else:
            sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
