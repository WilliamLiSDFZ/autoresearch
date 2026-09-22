"""CLI, immutable invocation records and source/result handoff for autoresearch."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "autoresearch-analogy-v1"


# 分块读取文件并计算 SHA256，避免一次加载大文件。
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# 读取 UTF-8 JSON，并拒绝 NaN 等非标准数值常量。
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON: {value}")))


# 将数据写成 UTF-8 JSON，可选择拒绝覆盖已有文件。
def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


# 返回带 UTC 时区信息的 ISO 格式时间戳。
def timestamp():
    return datetime.now(timezone.utc).isoformat()


# 检查写入路径与固定输入目录互不包含，防止污染数据或父实验。
def separate_from_inputs(destination, args):
    """Never write cache, freeze records or outputs into fixed input directories."""
    destination = Path(destination).expanduser().resolve()
    for key in ("prepared_dir", "corpus_dir", "parent_artifacts"):
        value = getattr(args, key, None)
        if value is not None:
            source = Path(value).expanduser().resolve()
            if destination == source or source in destination.parents or destination in source.parents:
                raise ValueError("Writable paths must be separate from corpus, prepared data and parent artifacts")
    for value in getattr(args, "reference_artifacts", None) or []:
        source = Path(value).expanduser().resolve()
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("Writable paths must be separate from reference artifacts")
    return destination


@dataclasses.dataclass
class AgentOptions:
    max_turns: int = 14
    top_k: int = 10
    max_mechanisms: int = 3
    report_char_budget: int = 0
    max_output_tokens: int = 16384

    def __post_init__(self):
        for field in dataclasses.fields(self):
            minimum = 0 if field.name == "report_char_budget" else 1
            if type(getattr(self, field.name)) is not int or getattr(self, field.name) < minimum:
                raise ValueError(f"agent.{field.name} must be an integer >= {minimum}")
        if self.top_k > 20:
            raise ValueError("agent.top_k must not exceed 20")


def _options(cls, value, *, defaults=None):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} configuration must be an object")
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**{**(defaults or {}), **value})


# 合并并校验命令行、JSON 和环境配置，构建检索及模型选项。
def settings(args, *, require_key=True):
    from .context import ContextOptions
    from .code_tools import CodeToolOptions
    from .fulltext import FullTextConfig, parser_versions
    from .responses import uses_responses, supports_reasoning_effort
    raw = read_json(args.config) if args.config else {}
    if not isinstance(raw, dict) or raw.keys() - {"agent", "context", "code_tools", "fulltext"}:
        raise ValueError("Config accepts only agent, context, code_tools and fulltext sections")
    agent = _options(AgentOptions, raw.get("agent", {}))
    context = _options(ContextOptions, raw.get("context", {}), defaults={"version": 2})
    code = _options(CodeToolOptions, raw.get("code_tools", {}))
    fulltext = _options(FullTextConfig, raw.get("fulltext", {}), defaults={"enabled": True})
    if context.version != 2 or not fulltext.enabled:
        raise ValueError("This implementation requires context v2 and full-text tools enabled")
    # Dataclass construction alone does not validate every inherited option.
    for obj in (context, code, fulltext):
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if field.name in {"enabled", "offline"}:
                if type(value) is not bool:
                    raise ValueError(f"{field.name} must be boolean")
            elif isinstance(field.default, (int, float)) and not isinstance(field.default, bool):
                zero_allowed = obj is context and (field.name.endswith("_chars") or field.name == "max_input_tokens")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
                        value) or (value < 0 if zero_allowed else value <= 0):
                    raise ValueError(f"{field.name} must be finite and {'nonnegative' if zero_allowed else 'positive'}")
                if isinstance(field.default, int) and not isinstance(value, int):
                    raise ValueError(f"{field.name} must be an integer")
    if args.cache_dir:
        fulltext.cache_dir = str(Path(args.cache_dir).expanduser().resolve())
    elif not fulltext.cache_dir:
        fulltext.cache_dir = str(ROOT / "results" / "analogy-cache" / "paper_fulltext")
    else:
        fulltext.cache_dir = str(Path(fulltext.cache_dir).expanduser().resolve())
    if args.offline:
        fulltext.offline = True
    separate_from_inputs(fulltext.cache_dir, args)
    missing = [key for key, value in parser_versions().items() if value == "unavailable"]
    if missing:
        raise ValueError(f"Missing full-text parser dependencies: {', '.join(missing)}")
    model = args.model or os.environ.get("ANALOGY_MODEL", "")
    if not model.strip():
        raise ValueError("Set ANALOGY_MODEL or --model explicitly")
    if uses_responses(model, args.api) and not supports_reasoning_effort(model, args.reasoning_effort):
        raise ValueError("Unsupported Responses reasoning effort for this model")
    base_url = args.base_url or os.environ.get("ANALOGY_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"https",
                                 "http"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API base URL must be HTTP(S), without credentials, query or fragment")
    api_key = os.environ.get("ANALOGY_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if require_key and not api_key:
        raise ValueError("Set ANALOGY_API_KEY or OPENAI_API_KEY; credentials are never read from config files")
    llm = SimpleNamespace(model=model, base_url=base_url, api_key=api_key,
                          api=args.api, reasoning_effort=args.reasoning_effort)
    return agent, context, code, fulltext, llm


# 汇总代码、输入、依赖及模型配置，生成实验协议的冻结记录。
def frozen_contract(args, settings_tuple):
    agent, context, code, fulltext, llm = settings_tuple
    from .context import load_prepared_manifest
    from .responses import uses_responses
    from .model_profiles import is_openai_reasoning_model
    response_api = uses_responses(llm.model, llm.api)
    prepared = load_prepared_manifest(args.prepared_dir)
    files = [ROOT / "analogy_agent.py", ROOT / "prepare.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    files += sorted((ROOT / "autoresearch_analogy").glob("*.py"))
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions[name.lower()] = dist.version
    return {
        "protocol": PROTOCOL,
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "task_sha256": sha256(args.task_file),
        "prepared_id": prepared["prepared_id"],
        "corpus_sha256": sha256(Path(args.corpus_dir) / "records.jsonl"),
        "corpus_manifest_sha256": sha256(Path(args.corpus_dir) / "manifest.json"),
        "python": list(sys.version_info[:3]), "dependencies": dict(sorted(versions.items())),
        "agent": dataclasses.asdict(agent), "context": dataclasses.asdict(context),
        "code_tools": dataclasses.asdict(code), "fulltext": dataclasses.asdict(fulltext),
        "model": {"model": llm.model, "api": llm.api, "base_url": llm.base_url,
                  "effective_api": "responses" if response_api else "chat_completions",
                  "requested_reasoning_effort": llm.reasoning_effort,
                  "effective_reasoning_effort": (llm.reasoning_effort if response_api else
                                                 "none" if is_openai_reasoning_model(llm.model) else None)},
        "declared_resources": args.resources,
    }


def _safe_error(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(secret, "[REDACTED]") if secret else message


# 执行预检或阶段检索，并保存报告、证据和调用状态。
def run_command(args):
    from .context import build_packet
    from .corpus import load_corpus
    from .observed_loop import run
    history_as_of = timestamp()
    opts = settings(args)
    agent, context, code, fulltext, llm = opts
    contract = frozen_contract(args, opts)
    if args.lock_file and args.command == "run" and read_json(args.lock_file) != contract:
        raise ValueError("Frozen protocol differs from --lock-file; do not modify it during a run")
    corpus = load_corpus(args.corpus_dir)
    packet, metadata, runtime, code_session = build_packet(
        args.stage, args.task_file, args.prepared_dir,
        parent_artifacts=args.parent_artifacts, history=args.history,
        resources=args.resources, options=context, code_options=code,
        reference_artifacts=args.reference_artifacts,
        history_run_dir=args.history_run_dir, history_as_of=history_as_of)
    if args.command == "preflight":
        if args.lock_file:
            separate_from_inputs(args.lock_file, args)
            Path(args.lock_file).parent.mkdir(parents=True, exist_ok=True)
            write_json(args.lock_file, contract, exclusive=True)
        print(json.dumps({"status": "ready", "stage": args.stage, "papers": len(corpus),
                          "prepared_id": contract["prepared_id"], "fulltext": True,
                          "offline_papers": fulltext.offline,
                          "model": llm.model, "model_contacted": False}, ensure_ascii=False))
        return 0
    output = separate_from_inputs(args.output_dir, args)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = getattr(code_session, "history_snapshot", None)
    if snapshot is not None:
        write_json(output / "history.json", snapshot, exclusive=True)
        metadata["history"]["snapshot_sha256"] = sha256(output / "history.json")
    write_json(output / "context.json", {"stage": args.stage, "packet": packet,
                                         "metadata": metadata, "runtime_context": runtime})
    manifest = {"protocol": PROTOCOL, "stage": args.stage, "call_id": output.name,
                "started_at": history_as_of, "status": "running", "frozen": contract,
                "input_metadata": metadata}
    write_json(output / "manifest.json", manifest)
    try:
        result = run(packet, corpus, llm, mode=args.stage, fulltext=fulltext,
                     context_options=context, code_session=code_session,
                     packet_metadata=metadata, runtime_context=runtime,
                     **dataclasses.asdict(agent))
        # Configuration, executable modules and corpus must still match after inference.
        if frozen_contract(args, settings(args)) != contract:
            result.delivery_status, result.failure_kind = "failed", "frozen_input_changed"
            result.reason = "Frozen inputs changed during retrieval; discard the report"
            result.report, result.report_md = None, ""
        status = ("ok" if result.delivery_status.startswith("accepted_") else
                  "abstained" if result.delivery_status == "abstained" else "failed")
        # SDK exception strings can contain server-provided text; redact known credentials
        # before persisting any trace, failure or metadata.
        payload = json.loads(json.dumps(dataclasses.asdict(result), ensure_ascii=False).replace(
            llm.api_key, "[REDACTED]"))
        report = payload["report"] or {"mechanisms": [], "failure_reason": payload["reason"]}
        write_json(output / "report.json", report)
        rendered = payload["report_md"] or (
            f"# Analogy retrieval: {status}\n\n{payload['reason']}\n")
        (output / "report.md").write_text(rendered, encoding="utf-8")
        with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:
            for event in payload["trace"]:
                stream.write(json.dumps({"event": event}, ensure_ascii=False) + "\n")
        for key in ("fulltext", "context", "code_reads", "model_calls", "submission_attempts"):
            filename = "context_budget.json" if key == "context" else f"{key}.json"
            write_json(output / filename, payload[key])
        manifest.update(status=status, delivery_status=result.delivery_status,
                        failure_kind=result.failure_kind, reason=payload["reason"],
                        finished_at=timestamp(), turns=result.turns, seconds=result.seconds,
                        input_tokens=result.in_tokens, output_tokens=result.out_tokens,
                        queries=payload["queries"], paper_ids=payload["paper_ids"])
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": status, "report": str(output / "report.md"),
                          "delivery_status": result.delivery_status}, ensure_ascii=False))
        return 1 if status == "failed" else 0
    except Exception as exc:
        manifest.update(status="failed", failure_kind="runtime", finished_at=timestamp(),
                        reason=_safe_error(exc, llm.api_key))
        write_json(output / "manifest.json", manifest)
        raise


# 在采纳前只读复核历史，保留旧调用快照并明确是否需要刷新检索。
def check_history_command(args):
    from .context import _parent_snapshot
    from .history import build_history, compare_snapshots

    as_of = timestamp()
    root = args.history_run_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.parent != root / "analogy":
        raise ValueError("Report must belong to history_run_dir/analogy")
    previous = read_json(report_dir / "history.json")
    manifest = read_json(report_dir / "manifest.json")
    expected = manifest.get("input_metadata", {}).get("history", {}).get("snapshot_sha256")
    if expected != sha256(report_dir / "history.json"):
        raise ValueError("Report history snapshot hash mismatch or missing binding")
    if manifest.get("status") not in {"ok", "abstained"}:
        raise ValueError("Only a successful or abstained report can be reviewed for adoption")
    node, record, _, _, _ = _parent_snapshot(
        args.parent_artifacts, previous["prepared_id"], previous["metric_version"], include_log=False)
    if args.parent_artifacts.expanduser().resolve() != root / "trials" / node["id"]:
        raise ValueError("Current parent must belong to history_run_dir/trials")
    current, _ = build_history(root, parent_id=node["id"], run_id=record["run_id"],
        prepared_id=previous["prepared_id"], metric_version=previous["metric_version"], as_of=as_of)
    changes = compare_snapshots(previous, current)
    changed_ids = set(changes["added_trial_ids"] + changes["changed_trial_ids"])
    before = {item["trial_id"]: item for item in previous["experiments"]}
    after = {item["trial_id"]: item for item in current["experiments"]}
    parent_source_changed = (before.get(node["id"], {}).get("source_sha256") != record["sha256"])
    refresh = (changes["parent_changed"] or parent_source_changed
               or current.get("best_trial_id") not in {None, node["id"]})
    review = bool(changed_ids or changes["removed_trial_ids"] or changes["best_changed"])
    status, code = ("refresh_required", 4) if refresh else (("review_required", 3) if review else ("unchanged", 0))
    result = {"status": status, "checked_at_utc": as_of, "run_id": previous["run_id"],
        "report_id": report_dir.name, "history_snapshot_sha256": expected,
        "previous_as_of_utc": previous["as_of_utc"], "current_as_of_utc": current["as_of_utc"],
        "parent_source_changed": parent_source_changed, **changes,
        "changed_experiments": [after[key] for key in sorted(changed_ids)],
        "current_source_hashes": current["provenance"]["source_hashes"],
        "instruction": ("Run a fresh improve call on the current best before editing." if refresh else
            "Review new results against the selected mechanism. Refresh if relevant; otherwise record the "
            "review and continuation reason in adoption.json." if review else
            "History is unchanged; record this check in adoption.json before editing.")}
    if args.output:
        output = args.output.expanduser().resolve()
        if not output.is_relative_to(root) or output.is_relative_to(root / "trials"):
            raise ValueError("History checks must be saved within the run, outside trial artifacts")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))
    return code


# 在训练前保存源码快照，并登记实验身份与固定数据版本。
def snapshot_command(args):
    from .context import load_prepared_manifest
    prepared = load_prepared_manifest(args.prepared_dir)
    source = Path(args.source).expanduser().resolve()
    content = source.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_id) or not args.run_id.strip():
        raise ValueError("Use a nonempty run ID and an alphanumeric/underscore/hyphen experiment ID")
    output = separate_from_inputs(args.artifact_dir, args)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.py").write_bytes(content)
    write_json(output / "source.json", {
        "protocol": PROTOCOL, "sha256": source_hash, "source_path": str(source),
        "experiment_id": args.experiment_id, "run_id": args.run_id,
        "prepared_id": prepared["prepared_id"], "execution_status": "pending",
        "created_at": timestamp(),
    })
    print(json.dumps({"status": "pending", "artifact_dir": str(output), "source_sha256": source_hash}))
    return 0


# 校验源码与结果后，将指标、配置和日志绑定到训练前快照。
def complete_command(args):
    from .context import load_prepared_manifest
    prepared = load_prepared_manifest(args.prepared_dir)
    directory = Path(args.artifact_dir).expanduser().resolve()
    receipt = read_json(directory / "source.json")
    if receipt.get("execution_status") != "pending":
        raise ValueError("Only a pending snapshot may be completed")
    if receipt.get("prepared_id") != prepared["prepared_id"]:
        raise ValueError("Snapshot uses a different prepared split")
    if sha256(directory / "source.py") != receipt["sha256"] or sha256(receipt["source_path"]) != receipt["sha256"]:
        raise ValueError("Source changed after snapshot; cannot bind this result to the snapshot")
    metrics = read_json(directory / "metrics.json")
    config = read_json(directory / "config.json")
    score = metrics.get("score")
    if metrics.get("prepared_id") != receipt["prepared_id"] or type(score) not in (int, float) or not math.isfinite(
            score):
        raise ValueError("Metrics need a finite score and the matching prepared_id")
    if metrics.get("maximize") is not True or not 0 <= score <= 1 or not isinstance(config, dict):
        raise ValueError("Invalid Jigsaw result/configuration")
    if metrics.get("metric_version") != prepared["metric_version"]:
        raise ValueError("Metrics use a different metric version")
    receipt.update(metrics_sha256=sha256(directory / "metrics.json"),
                   config_sha256=sha256(directory / "config.json"))
    if args.log_file:
        content = Path(args.log_file).read_bytes()
        log = directory / "run.log"
        if log.exists() and log.read_bytes() != content:
            raise ValueError("A different run.log already exists in this experiment directory")
        log.write_bytes(content)
    if (directory / "run.log").exists():
        receipt["log_sha256"] = sha256(directory / "run.log")
    receipt.update(execution_status="completed", completed_at=timestamp())
    # Update the receipt only after every required source/result check passed.
    temporary = directory / "source.json.tmp"
    write_json(temporary, receipt, exclusive=True)
    temporary.replace(directory / "source.json")
    print(json.dumps({"status": "completed", "artifact_dir": str(directory), "score": score}))
    return 0


# 定义检索、预检、快照和结果绑定四个子命令的参数。
def parser():
    p = argparse.ArgumentParser(description="Read-only MLEvolve-style analogy retrieval, including full papers")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight"):
        cmd = commands.add_parser(name,
                                  help="Retrieve a report" if name == "run" else "Validate configuration without an API call")
        cmd.add_argument("--stage", choices=("draft", "improve"), required=True)
        cmd.add_argument("--task-file", type=Path, required=True)
        cmd.add_argument("--prepared-dir", type=Path, required=True)
        cmd.add_argument("--corpus-dir", type=Path, required=True)
        cmd.add_argument("--parent-artifacts", type=Path)
        cmd.add_argument("--reference-artifacts", type=Path, action="append", default=[],
                         help="Completed same-run historical candidate; repeat to allow code comparisons")
        cmd.add_argument("--history", type=Path)
        cmd.add_argument("--history-run-dir", type=Path,
                         help="Freeze same-run trials, results and adoption history before improve")
        cmd.add_argument("--resources", default="Unknown; no experiment wall-clock deadline is configured.")
        cmd.add_argument("--config", type=Path, help="Non-secret JSON settings, frozen for a run")
        cmd.add_argument("--model")
        cmd.add_argument("--base-url")
        cmd.add_argument("--api", choices=("auto", "responses", "chat_completions"), default="auto")
        cmd.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                         default="high")
        cmd.add_argument("--cache-dir", type=Path)
        cmd.add_argument("--offline", action="store_true", help="PDF cache only; LLM requests still use the API")
        cmd.add_argument("--lock-file", type=Path, help="Preflight creates this freeze record; run checks it")
        if name == "run":
            cmd.add_argument("--output-dir", type=Path, required=True)
        cmd.set_defaults(handler=run_command)
    snap = commands.add_parser("snapshot", help="Save source BEFORE running a candidate; never executes it")
    snap.add_argument("--source", type=Path, required=True)
    snap.add_argument("--artifact-dir", type=Path, required=True)
    snap.add_argument("--prepared-dir", type=Path, required=True)
    snap.add_argument("--experiment-id", required=True)
    snap.add_argument("--run-id", required=True)
    snap.set_defaults(handler=snapshot_command)
    complete = commands.add_parser("complete", help="Bind a finished result to its pre-execution snapshot")
    complete.add_argument("--artifact-dir", type=Path, required=True)
    complete.add_argument("--prepared-dir", type=Path, required=True)
    complete.add_argument("--log-file", type=Path)
    complete.set_defaults(handler=complete_command)
    check = commands.add_parser("check-history", help="Review new results before adopting a prefetched improve report")
    check.add_argument("--history-run-dir", type=Path, required=True)
    check.add_argument("--report-dir", type=Path, required=True)
    check.add_argument("--parent-artifacts", type=Path, required=True,
                       help="Current best finalized candidate, not necessarily the report's original parent")
    check.add_argument("--output", type=Path, help="New same-run JSON check receipt; never overwritten")
    check.set_defaults(handler=check_history_command)
    return p


# 解析命令并分发执行，将异常转换为脱敏错误信息和退出码。
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Retain the --stage ... invocation form from the original integration proposal.
    if argv and argv[0].startswith("--") and argv[0] not in {"--help", "--version"}:
        argv.insert(0, "run")
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:
        secret = os.environ.get("ANALOGY_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        print("analogy_agent: " + _safe_error(exc, secret), file=sys.stderr)
        return 2
