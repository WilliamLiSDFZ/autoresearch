#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.2", "matplotlib>=3.8", "openai>=2.0", "sentence-transformers>=5.0"]
# ///
"""Compare complete autoresearch solutions or evidenced changes using cosine Vendi."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import textwrap


ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
COHORT_FIELDS = ("study_id", "task", "task_hash", "evaluator_sha256", "stage", "view")


def digest(value):
    """Stable identity for settings, source packets and cached representations."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    """Publish a complete JSON file, without leaving a partial cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fallback=("run_id", "issue")):
    """Keep audit tables readable even when no candidates can be scored."""
    columns = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_samples(path):
    """Import normalized mechanism cards/vectors, never arbitrary code as a summary."""
    from autoresearch_vendi.assessment import preserve_assessment_status
    unique = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("task", "study_id", "run_id", "arm", "candidate_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Input line {number}: missing string {key}")
        if row.get("stage") not in ("", "draft", "improve", "solution"):
            raise ValueError(f"Input line {number}: invalid stage")
        if row["stage"] == "solution" and row.get("assessment_status"):
            raise ValueError("Whole solutions must not carry diff assessment statuses")
        row.setdefault("view", "implementation")
        row.setdefault("pair_id", "")
        row.setdefault("parent_id", "")
        row.setdefault("text", "")
        row.setdefault("representation_version", "external-normalized-v1")
        row.setdefault("source_refs", [])
        if row["view"] != "implementation" or not isinstance(row["text"], str):
            raise ValueError("Only normalized implementation text is supported")
        if "embedding" in row and not isinstance(row.get("embedding_model"), str):
            raise ValueError("Every supplied vector needs an embedding_model identity")
        excluded = row.get("extraction_status") == "excluded"
        nonscoring = preserve_assessment_status(row)
        if excluded:
            row["extraction_status"] = "excluded"
        elif not nonscoring:
            row["extraction_status"] = ("ok" if row["text"].strip() or "embedding" in row
                                        else row.get("extraction_status", "missing_source"))
            if row["extraction_status"] == "ok":
                row.pop("error", None)
        if not row["stage"] and not excluded:
            if row["text"].strip() or "embedding" in row:
                raise ValueError("Unknown-stage candidates cannot contain scoring representations")
            row.update(extraction_status="missing_source", error=row.get("error") or "unknown_stage")
        elif row.get("extraction_status") == "pending":
            row.update(extraction_status="missing_source", error="rerun_from_sources_for_extraction")
        key = tuple(row[k] for k in ("study_id", "task", "run_id", "candidate_id", "view"))
        if key in unique and unique[key] != row:
            raise ValueError(f"Conflicting duplicate candidate: {key}")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def validate_samples(samples):
    """Reject mixed measurement settings and ambiguous pairings before model calls."""
    versions, runs, pair_runs, pair_sources = defaultdict(set), {}, {}, defaultdict(dict)
    for row in samples:
        for field in ("task", "study_id", "run_id", "arm", "candidate_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Missing candidate identity: " + field + "; supply a manifest")
        group = tuple(row.get(key, "") for key in COHORT_FIELDS)
        versions[group].add((row.get("representation_version", ""), row.get("summary_model", ""),
                             row.get("summary_prompt_sha256", "")))
        run_key = (row.get("study_id", ""), row["task"], row["run_id"])
        identity = (row["arm"], row.get("pair_id", ""), row.get("prepared_id", ""))
        if run_key in runs and runs[run_key] != identity:
            raise ValueError("Inconsistent metadata for run " + row["run_id"])
        runs[run_key] = identity
        if row.get("pair_id"):
            pair_key = (row.get("study_id", ""), row["task"], row["pair_id"], row["arm"])
            if pair_key in pair_runs and pair_runs[pair_key] != row["run_id"]:
                raise ValueError("Multiple runs for one study/task/pair/arm; use a manifest")
            pair_runs[pair_key] = row["run_id"]
            pair_sources[pair_key[:-1]][row["run_id"]] = row
    for group, members in pair_sources.items():
        for field in ("prepared_id", "task_hash", "evaluator_sha256", "metric_version", "maximize"):
            values = {json.dumps(row[field], sort_keys=True) for row in members.values() if row.get(field) not in (None, "")}
            if len(values) > 1:
                raise ValueError(f"Incompatible {field} within pair {group}")
    for group, found in versions.items():
        # Failed/missing representations need no model label; successful ones must agree.
        representations = {item[0] for item in found}
        models = {item[1:] for item in found if item[1] or item[2]}
        if len(representations) > 1 or len(models) > 1:
            raise ValueError(f"Mixed representation or extraction versions in {group}; re-extract consistently")


def coverage_rows(samples, issues):
    """Count all attempts, including unmeasurable changes and failed training."""
    groups = defaultdict(list)
    for row in samples:
        groups[tuple(row.get(k, "") for k in ("study_id", "task", "run_id", "arm", "stage", "view"))].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        counts = Counter(row.get("extraction_status", "pending") for row in rows)
        changed = sum(row.get("assessment_status") == "changed" for row in rows)
        result.append(dict(zip(("study_id", "task", "run_id", "arm", "stage", "view"), key)) | {
            "pair_id": rows[0].get("pair_id", ""), "n_candidates": len(rows),
            "n_available": counts["ok"], "n_valid": sum(row.get("is_valid") is True for row in rows),
            "n_not_verified_completed": sum(row.get("is_valid") is not True for row in rows),
            "n_changed": changed, "n_no_change": counts["no_change"],
            "n_insufficient": counts["insufficient_evidence"], "n_missing": counts["missing_source"],
            "n_errors": counts["error"], "n_pending": counts["pending"],
            "n_excluded": counts["excluded"],
            "changed_fraction": changed / len(rows),
            "no_change_fraction": counts["no_change"] / len(rows),
            "insufficient_fraction": counts["insufficient_evidence"] / len(rows),
            "issues": ";".join(sorted({row.get("error", "") for row in rows if row.get("error")}))})
    for row in result:
        if row["stage"] == "solution":
            for field in ("n_changed", "n_no_change", "n_insufficient", "changed_fraction",
                          "no_change_fraction", "insufficient_fraction"):
                row.pop(field)
    return result + issues


def score_samples(samples, baseline, repeats, seed):
    """Apply the unchanged AKB kernel within isolated study/task/representation cohorts."""
    from autoresearch_vendi.metrics import compare_samples
    ready = [row for row in samples if row.get("extraction_status") == "ok"]
    cohorts = defaultdict(list)
    for row in ready:
        cohorts[tuple(row.get(k, "") for k in COHORT_FIELDS)].append(row)
    scores, comparisons = [], []
    for key, rows in sorted(cohorts.items()):
        models = {row.get("embedding_model") for row in rows}
        if None in models or len(models) != 1:
            raise ValueError("A comparison must use exactly one embedding configuration")
        cohort = digest(key)[:12]
        # The legacy helper groups on task/stage/view; call it once per full cohort.
        local_scores, local_comparisons = compare_samples(rows, baseline=baseline, repeats=repeats, seed=seed)
        metadata = {"study_id": key[0], "cohort": cohort}
        scores.extend(metadata | row for row in local_scores)
        comparisons.extend(metadata | row for row in local_comparisons)
    return scores, comparisons


def full_solution_scores(samples):
    """Describe each complete run separately; unequal-size scores are not paired effects."""
    from autoresearch_vendi.metrics import vendi_score
    groups = defaultdict(list)
    for row in samples:
        if row.get("extraction_status") == "ok":
            groups[tuple(row.get(k, "") for k in COHORT_FIELDS) + (row["run_id"],)].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        first = rows[0]
        result.append({"study_id": first["study_id"], "task": first["task"], "run_id": first["run_id"],
                       "arm": first["arm"], "pair_id": first.get("pair_id", ""),
                       "cohort": digest(key[:-1])[:12], "stage": "solution", "n": len(rows),
                       "vendi": vendi_score([row["embedding"] for row in rows]),
                       "status": "descriptive_unmatched_counts"})
    return result


def extract_samples(samples, out, summarizer):
    """Save source evidence, then extract only neutral implementation changes."""
    from autoresearch_vendi.changes import build_change_packet
    from autoresearch_vendi.assessment import CHANGE_PROMPT, assessment_text
    from autoresearch_vendi.runtime import FIELDS, SUMMARY_PROMPT, SOLUTION_PROMPT
    for index, row in enumerate(samples, 1):
        if row.get("extraction_status") != "pending":
            continue
        print(f"[{index}/{len(samples)}] {row['run_id']} / {row['candidate_id']}", flush=True)
        try:
            if row["stage"] == "solution":
                source = "\n".join(f"SOURCE:{i}: {line}" for i, line in enumerate(row["source"].splitlines(), 1))
                card, chunks = summarizer.summarize_solution(source)
                row.update(text=card["summary"], solution_title=card["title"], mechanism_card=card,
                           source_chunks=chunks, extraction_status="ok", summary_model=summarizer.model,
                           summary_prompt_sha256=digest(SOLUTION_PROMPT))
            elif row["stage"] == "improve":
                if not row.get("parent_source"):
                    row.update(extraction_status="missing_source", error="parent_code_unavailable")
                    continue
                packet = build_change_packet(row["parent_source"], row["source"], max_chars=160000)
                name = digest([row["run_id"], row["candidate_id"], packet["source_hashes"]])[:24] + ".json"
                write_json(out / "change_evidence" / name, packet)
                row["change_evidence_file"] = "change_evidence/" + name
                card = summarizer.assess_change(packet)
                row.update(text=assessment_text(card), mechanism_card=card, assessment_status=card["status"],
                           extraction_status="ok" if card["status"] == "changed" else card["status"],
                           summary_model=summarizer.model, summary_prompt_sha256=digest(CHANGE_PROMPT))
            elif row["stage"] == "draft":
                source = "\n".join(f"CHILD:{i}: {line}" for i, line in enumerate(row["source"].splitlines(), 1))
                card, chunks = summarizer.summarize(source, "implementation")
                row.update(text="; ".join(f"{k}: {card[k]}" for k in FIELDS if card[k]),
                           mechanism_card=card, source_chunks=chunks, extraction_status="ok",
                           summary_model=summarizer.model, summary_prompt_sha256=digest(SUMMARY_PROMPT))
            else:
                row.update(extraction_status="missing_source", error="unknown_stage")
        except Exception as exc:
            # Source/API errors are recorded without provider payloads or credentials.
            reason = "missing_summary_api_key" if type(exc).__name__ == "_ConfigurationError" else "extraction_failed:" + type(exc).__name__
            row.update(extraction_status="error", error=reason)


def plot_results(out, scores, comparisons, coverage, issues):
    """Draw diversity curves and paired effects without implying quality/significance."""
    if not scores:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    output_files = []
    for cohort in sorted({row["cohort"] for row in scores}):
        rows = [row for row in scores if row["cohort"] == cohort]
        first = rows[0]
        title = textwrap.fill(f"{first['task']} | {first['study_id']} | {first['stage']}", 95)
        prefix = re.sub(r"[^\w.-]", "-", first["task"] + "__" + first["study_id"])[:85] + "__" + first["stage"] + "__" + cohort
        qualification = "Conditional on measurable static changes." if first["stage"] == "improve" else "Static implementation summaries."
        note = (qualification + " Bands show candidate-subset variation, not effect confidence intervals.\n"
                "More diversity does not imply better performance. Read coverage.csv and comparability.csv for missing evidence and run differences.")

        def save(fig, name):
            fig.text(0.07, 0.015, note, fontsize=8, color="#64748b", va="bottom")
            fig.tight_layout(rect=(0, 0.11, 1, 1))
            file = out / f"{prefix}_{name}.png"
            fig.savefig(file, dpi=170, bbox_inches="tight")
            output_files.append(file.name)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 5.5))
        colors = {arm: plt.get_cmap("tab10")(i % 10) for i, arm in enumerate(sorted({r["arm"] for r in rows}))}
        for run_id in sorted({r["run_id"] for r in rows}):
            selected = sorted((r for r in rows if r["run_id"] == run_id), key=lambda r: r["m"])
            row = selected[0]
            label = f"{row['arm']} | {row['pair_id'] or run_id} (n={row['n_total']})"
            xs = [r["m"] for r in selected]
            ax.plot(xs, [r["vendi"] for r in selected], "o-", color=colors[row["arm"]], label=label)
            ax.fill_between(xs, [r["subset_low"] for r in selected], [r["subset_high"] for r in selected], alpha=0.12, color=colors[row["arm"]])
        ax.set(title=title, xlabel="Candidates per run (matched m)", ylabel="Vendi score (q=1)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        save(fig, "vendi")

        paired = [r for r in comparisons if r["cohort"] == cohort and r["kind"] == "pair" and r["delta"] is not None]
        if not paired:
            continue
        fig, ax = plt.subplots(figsize=(12, 5.5))
        keys = sorted({(r["arm"], r["pair_id"]) for r in paired})
        for arm, pair_id in keys:
            selected = sorted((r for r in paired if (r["arm"], r["pair_id"]) == (arm, pair_id)), key=lambda r: r["m"])
            ax.plot([r["m"] for r in selected], [r["delta"] for r in selected], "o-", label=f"{pair_id}: {arm} vs {selected[0]['baseline']}")
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Candidates per run (matched m)", ylabel="Vendi difference (positive = more diverse)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        save(fig, "effect")

        # The largest shared m is declared in the title; it never changes the cohort.
        largest = max(r["m"] for r in paired)
        selected = [r for r in paired if r["m"] == largest]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        first_arm = selected[0]["baseline"]
        arms = [first_arm] + sorted({r["arm"] for r in selected} - {first_arm})
        draws = defaultdict(dict)
        for row in selected:
            draws[row["pair_id"]].update({row["baseline"]: row["baseline_vendi"], row["arm"]: row["arm_vendi"]})
        for pair_id, values in sorted(draws.items()):
            xs = [i for i, arm in enumerate(arms) if arm in values]
            ax.plot(xs, [values[arms[i]] for i in xs], "o-", label=pair_id)
        ax.set_xticks(range(len(arms)), arms)
        ax.set(title=title + f"\nPaired Vendi at matched m={largest}", ylabel="Vendi score (q=1)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        save(fig, "paired")
    return output_files


def comparability_rows(samples):
    """Expose known differences and unknown budgets without claiming controlled effects."""
    groups = defaultdict(dict)
    for row in samples:
        if row.get("pair_id"):
            groups[(row.get("study_id", ""), row["task"], row["pair_id"])][row["run_id"]] = row
    result = []
    fields = ("gpu", "start_commit", "seed", "budget_seconds", "agent_model", "prepared_id", "evaluator_sha256")
    for (study, task, pair), runs in sorted(groups.items()):
        notes = []
        for field in fields:
            values = [row.get(field) for row in runs.values()]
            if any(value in (None, "") for value in values):
                notes.append(field + "_unknown")
            elif len({json.dumps(value, sort_keys=True) for value in values}) > 1:
                notes.append(field + "_differs")
        result.append({"study_id": study, "task": task, "pair_id": pair,
                       "runs": ";".join(sorted(runs)), "warnings": ";".join(notes)})
    return result


def save_outputs(out, samples, issues, parents, scores, comparisons, settings, identity, calls,
                 prepare_only=False, no_plots=False):
    """Persist evidence coverage even when there is no scoreable comparison."""
    out.mkdir(parents=True, exist_ok=True)
    coverage = coverage_rows(samples, issues)
    comparability = comparability_rows(samples)
    solutions = settings.get("mode") == "solutions"
    write_csv(out / "coverage.csv", coverage)
    write_csv(out / "comparability.csv", comparability, ("study_id", "task", "pair_id", "warnings"))
    if solutions:
        cards = [{"run_id": row["run_id"], "candidate_id": row["candidate_id"], "arm": row["arm"],
                  "source_sha256": row.get("source_hash", ""), "is_valid": row.get("is_valid", False),
                  "status": row.get("extraction_status", ""), "title": row.get("solution_title", ""),
                  "summary": row.get("text", ""),
                  "evidence": json.dumps(row.get("mechanism_card", {}).get("evidence", [])),
                  "error": row.get("error", "")} for row in samples]
        write_csv(out / "solution_cards.csv", cards, ("run_id", "candidate_id", "title", "summary", "evidence", "status"))
        write_csv(out / "full_run_scores.csv", [] if prepare_only else full_solution_scores(samples),
                  ("study_id", "task", "run_id", "arm", "n", "vendi", "status"))
    else:
        write_csv(out / "parent_map.csv", parents,
                  ("run", "candidate_id", "stage", "parent_id", "child_sha256", "parent_sha256", "evidence", "status"))
    write_csv(out / "run_scores.csv", scores, ("study_id", "task", "run_id", "stage", "m", "vendi"))
    write_csv(out / "comparisons.csv", comparisons, ("study_id", "task", "arm", "baseline", "m", "delta", "status"))
    with (out / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            # Raw source is needed only by extraction; its evidence packet is exported separately.
            exported = {k: v for k, v in row.items() if k not in ("source", "parent_source", "change_packet")}
            stream.write(json.dumps(exported, ensure_ascii=False, allow_nan=False) + "\n")
    manifest_file = out / "manifest.json"
    old_charts = json.loads(manifest_file.read_text()).get("charts", []) if manifest_file.exists() else []
    charts = [] if prepare_only or no_plots else plot_results(out, scores, comparisons, coverage, issues)
    for filename in set(old_charts) - set(charts):
        path = out / filename
        if path.parent == out and path.suffix in (".png", ".pdf"):
            path.unlink(missing_ok=True)
    source_files = [Path(__file__), ROOT / "analyze_runs.py", *sorted((ROOT / "autoresearch_vendi").glob("*.py"))]
    if solutions:
        source_files.append(ROOT / "compare_solution_vendi.py")
    measurement = ("whole-candidate static solution summaries; no parent required" if solutions
                   else "conditional on evidenced static changes")
    manifest = {"version": "autoresearch-vendi-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "settings": settings, "embedding": identity, "summary_api_calls": calls,
                "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "charts": charts, "n_samples": len(samples), "prepare_only": prepare_only,
                "comparison": "q=1 cosine Vendi; matched counts; " + measurement + "; descriptive"}
    write_json(manifest_file, manifest)
    scope = ("Complete source snapshots are summarized independently using one neutral prompt. Initial and later "
             "candidates share stage=solution; parent links, retrieval reports and performance scores are not summary inputs. "
             "Repeated solutions retain their frequency. This measures diversity of attempted implementations, not all "
             "proposed ideas, literature novelty or successful execution. Full-run scores use unequal counts and are "
             "descriptive; use matched-count scores for comparisons." if solutions else
             "Implementation diversity is conditional on evidenced changes. Read changed/no-change/insufficient/missing "
             "counts and fractions alongside Vendi. Identical mechanisms in different candidates retain their frequency. "
             "Missing parents are never replaced by the previous trial or nearest source. Evidence checks establish "
             "source provenance and static connections, not semantic correctness or scientific novelty.")
    report = ["# Autoresearch " + ("solution " if solutions else "") + "Vendi analysis", "",
              "Mode: " + ("inventory only; no models called" if prepare_only else measurement), "",
              f"Candidates: {len(samples)}; available representations: {sum(r.get('extraction_status') == 'ok' for r in samples)}; run issues: {len(issues)}.", "",
              "Each study/task/source-protocol/stage/view is separate. A fixed cohort of runs with at least two "
              "available candidates shares m=2..minimum candidate count. An unavailable arm never contributes zero. "
              "Runs with one available candidate have no matched-count comparison. Failed training candidates are retained "
              "unless valid-only is selected; static code is not proof of successful runtime activation.", "",
              scope, "",
              "Effects = comparison-arm Vendi minus reference Vendi. Positive means more diverse, not better. "
              "Subset bands are candidate-subset variability, NOT effect confidence intervals. Explicit experimental pair "
              "identity is required; training seeds and timestamp proximity do not create pairs. More independent, "
              "matched experiments are required for causal/significance claims. Budget is not automatically equalized.", "",
              "## Coverage", ""]
    if solutions:
        report += ["| Run | Attempts | Available | Verified completed | Pending extraction | Missing / errors |",
                   "|---|---:|---:|---:|---:|---:|"]
    else:
        report += ["| Run | Stage | Attempts | Available | Changed | No change | Insufficient | Missing / errors |",
                   "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in coverage:
        if "n_candidates" in row:
            if solutions:
                report.append(f"| {row['run_id']} | {row['n_candidates']} | {row['n_available']} | "
                              f"{row['n_valid']} | {row['n_pending']} | {row['n_missing']} / {row['n_errors']} |")
            else:
                report.append(f"| {row['run_id']} | {row['stage']} | {row['n_candidates']} | {row['n_available']} | "
                              f"{row['n_changed']} | {row['n_no_change']} | {row['n_insufficient']} | {row['n_missing']} / {row['n_errors']} |")
    report += ["", "## Comparability", ""]
    report.extend(f"- {row['pair_id']}: {row['warnings'] or 'no recorded mismatch'}." for row in comparability)
    audit = "solution_cards.csv for source-grounded summaries" if solutions else "parent_map.csv for ancestry inventory (including unresolved rows)"
    report += ["", "See coverage.csv for source and extraction failures, " + audit + ", "
               "and manifest.json for measurement settings. Generic JSONL imports are already prepared representations; "
               "they are not revalidated against original source. Use a separate output directory for sensitivity analyses.", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main(argv=None, *, default_mode="changes"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("changes", "solutions"), default=default_mode,
                        help="changes requires ancestry; solutions compares complete source snapshots")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--runs", type=Path, help="downloaded run directories (default: ~/nautilus/autoresearch-result)")
    source.add_argument("--input", type=Path, help="saved normalized samples.jsonl; does not call a summary model")
    parser.add_argument("--manifest", type=Path, help="same run identity/exclusion CSV as analyze_runs.py")
    parser.add_argument("--parent-map", type=Path, help="optional reviewed ancestry overrides; do not pass the generated inventory CSV")
    parser.add_argument("--out", type=Path, help="default: results/vendi or results/vendi-solutions by mode")
    parser.add_argument("--cache", type=Path, help="separate cache directory; defaults by mode")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--stages", nargs="+", choices=("draft", "improve"), help="changes mode only")
    parser.add_argument("--valid-only", action="store_true", help="sensitivity analysis on verified completed trials only")
    parser.add_argument("--summary-model", default="gpt-5.6-terra")
    parser.add_argument("--summary-api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--base-url", help="overrides VENDI_BASE_URL / ANALOGY_BASE_URL / OPENAI_BASE_URL / LLM_BASE_URL")
    parser.add_argument("--embedding-backend", choices=("local", "openai"),
                        help="default: local for changes, openai for solutions")
    parser.add_argument("--embedding-model", help="default: pinned MiniLM locally or text-embedding-3-small via API")
    parser.add_argument("--embedding-base-url", help="override VENDI_EMBEDDING_BASE_URL; otherwise use the summary endpoint")
    parser.add_argument("--embedding-revision", help="default MiniLM uses the revision pinned by the AKB analysis")
    parser.add_argument("--embedding-max-length", type=int, help="text embedding window; default 512 for the pinned MiniLM")
    parser.add_argument("--reembed", action="store_true", help="replace all saved scoring vectors using one embedding configuration")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="stdlib inventory only: no API, downloads or output files")
    parser.add_argument("--prepare-only", action="store_true", help="write inventory/parent map/coverage without any model calls")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    solutions = args.mode == "solutions"
    if solutions and (args.parent_map or args.stages):
        parser.error("solutions mode uses every complete candidate and needs neither --parent-map nor --stages")
    args.stages = ["solution"] if solutions else (args.stages or ["draft", "improve"])
    args.embedding_backend = args.embedding_backend or ("openai" if solutions else "local")
    args.embedding_model = args.embedding_model or ("text-embedding-3-small" if args.embedding_backend == "openai" else EMBEDDING_MODEL)
    if args.embedding_backend == "openai" and (args.embedding_revision or args.embedding_max_length):
        parser.error("--embedding-revision/--embedding-max-length apply only to the local embedding backend")
    output_name = "vendi-solutions" if solutions else "vendi"
    args.out = args.out or ROOT / "results" / output_name
    args.cache = args.cache or ROOT / "results" / (output_name + "-cache")
    if args.repeats < 1 or (args.embedding_max_length is not None and args.embedding_max_length < 8):
        parser.error("--repeats must be positive; embedding window must be at least 8")
    if args.input and (args.manifest or args.parent_map):
        parser.error("--manifest/--parent-map apply to --runs; saved samples carry their metadata")
    if args.reembed and not args.input:
        parser.error("--reembed requires --input")
    args.out, args.cache = args.out.expanduser().resolve(), args.cache.expanduser().resolve()
    if not args.dry_run and args.parent_map and args.parent_map.expanduser().resolve() == args.out / "parent_map.csv":
        parser.error("--parent-map points to the generated parent_map.csv inventory, which may contain unresolved rows "
                     "and would be overwritten. Omit --parent-map to use recorded evidence automatically. "
                     "For reviewed overrides, copy only evidence-backed rows to a separate file outside --out.")
    if args.input:
        args.input = args.input.expanduser().resolve()
        samples, issues, parents = load_samples(args.input), [], []
    else:
        from autoresearch_vendi.reader import load_candidates
        args.runs = (args.runs or Path.home() / "nautilus" / "autoresearch-result").expanduser().resolve()
        samples, issues, parents = load_candidates(args.runs, args.manifest, args.parent_map, mode=args.mode)
    if any((row["stage"] == "solution") != solutions for row in samples):
        parser.error("Input representation does not match --mode; solution and change samples must stay separate")
    if args.arms:
        samples = [s for s in samples if s["arm"] in args.arms]
        issues = [s for s in issues if not s.get("arm") or s["arm"] in args.arms]
    for row in samples:
        if row.get("stage") and row["stage"] not in args.stages:
            row.update(extraction_status="excluded", error="stage_not_selected")
        elif args.valid_only and row.get("is_valid") is not True:
            row.update(extraction_status="excluded", error="excluded_valid_only")
    validate_samples(samples)
    inventory = {"candidates": len(samples), "runs": len({s["run_id"] for s in samples}),
                 "counts": dict(Counter(f"{r['arm']}/{r['stage']}/{r.get('extraction_status')}" for r in samples)),
                 "pending_extractions_before_cache": sum(s.get("extraction_status") == "pending" for s in samples),
                 "issues": issues, "parents": parents}
    if args.dry_run:
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return 0
    # An output path may live next to downloads, but never replace a raw run's summary.
    if any((p / "run.json").exists() for directory in (args.out, args.cache) for p in (directory, *directory.parents)) or (args.runs and args.out == args.runs):
        parser.error("Choose an analysis output directory separate from raw run artifacts")
    if args.input and args.input == args.out / "samples.jsonl":
        parser.error("Use a new --out directory; never overwrite the input samples.jsonl")
    existing_manifest = args.out / "manifest.json"
    if existing_manifest.exists():
        previous_mode = json.loads(existing_manifest.read_text()).get("settings", {}).get("mode", "changes")
        if previous_mode != args.mode:
            parser.error("Choose a separate --out directory for solution versus change analysis")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ("base_url", "embedding_base_url")}
    identity, calls = {"backend": "none"}, 0
    if not args.prepare_only:
        from autoresearch_vendi.runtime import (EmbeddingRequestError, Summarizer,
                                                embed_samples, prepare_reembedding)
        args.out.mkdir(parents=True, exist_ok=True)
        base_url = args.base_url or next((os.environ[k] for k in ("VENDI_BASE_URL", "ANALOGY_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL") if os.environ.get(k)), None)
        api_key = next((os.environ[k] for k in ("VENDI_API_KEY", "ANALOGY_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY") if os.environ.get(k)), None)
        if args.input:
            if args.reembed:
                prepare_reembedding(samples)
        else:
            summarizer = Summarizer(args.cache, model=args.summary_model, api=args.summary_api, base_url=base_url, api_key=api_key)
            settings["endpoint_sha256"] = digest(base_url)
            extract_samples(samples, args.out, summarizer)
            calls = summarizer.calls
        validate_samples(samples)
        revision = args.embedding_revision or (EMBEDDING_REVISION if args.embedding_model == EMBEDDING_MODEL else None)
        max_length = args.embedding_max_length
        if max_length is None and args.embedding_model == EMBEDDING_MODEL and not any("embedding" in s for s in samples if s.get("extraction_status") == "ok"):
            max_length = 512
        try:
            if args.embedding_backend == "openai":
                from autoresearch_vendi.runtime import embed_solution_samples
                embedding_url = args.embedding_base_url or os.environ.get("VENDI_EMBEDDING_BASE_URL") or base_url
                embedding_key = os.environ.get("VENDI_EMBEDDING_API_KEY") or api_key
                identity = embed_solution_samples(samples, args.cache, model_name=args.embedding_model,
                                                  base_url=embedding_url, api_key=embedding_key)
            else:
                identity = embed_samples(samples, args.cache, args.embedding_model, revision=revision, max_length=max_length)
        except Exception as exc:
            # Preserve expensive extracted texts and failure coverage if the encoder cannot run.
            reason = "missing_embedding_api_key" if type(exc).__name__ == "_ConfigurationError" else "embedding_failed:" + type(exc).__name__
            identity = {"backend": "failed", "error": reason}
            if isinstance(exc, EmbeddingRequestError):
                identity["diagnostic"] = str(exc)
                print("Embedding error: " + str(exc), file=sys.stderr)
                if exc.status_code == 404:
                    print("The configured endpoint/model did not serve the embedding request. "
                          "Set --embedding-base-url and VENDI_EMBEDDING_API_KEY for an embedding provider, "
                          "or explicitly choose --embedding-backend local. "
                          "Reuse saved summaries with --input and a new --out directory.", file=sys.stderr)
            for row in samples:
                if row.get("extraction_status") == "ok":
                    row.update(extraction_status="error", error=identity["error"])
                    row.pop("embedding", None)
        scores, comparisons = score_samples(samples, args.baseline, args.repeats, args.seed)
    else:
        scores, comparisons = [], []
    save_outputs(args.out, samples, issues, parents, scores, comparisons, settings, identity, calls,
                 prepare_only=args.prepare_only, no_plots=args.no_plots)
    print(f"Saved {len(samples)} candidates and {len(scores)} score rows to {args.out}; summary API calls: {calls}")
    failures = Counter(s.get("error", s.get("extraction_status")) for s in samples
                       if s.get("extraction_status") not in ("ok", "no_change", "excluded"))
    if failures:
        print("Coverage gaps: " + "; ".join(f"{key}={count}" for key, count in failures.items()))
    if args.prepare_only:
        print("Inventory only. Complete solution snapshots are ready for extraction; no parent map is needed." if solutions else
              "Inventory only. parent_map.csv includes unresolved rows; --parent-map is optional and accepts only reviewed overrides. Original snapshots are unchanged.")
        return 0
    all_no_change = any(s.get("extraction_status") == "no_change" for s in samples) and all(
        s.get("extraction_status") in ("no_change", "excluded") for s in samples)
    return 2 if failures or (not scores and not all_no_change) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit(f"compare_vendi: {exc}") from exc
