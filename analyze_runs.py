#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "scipy>=1.11"]
# ///
"""Analyze downloaded autoresearch runs; write paired/effect figures and audit tables."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import textwrap


ROOT = Path(__file__).resolve().parent
UUID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
OVERRIDE_FIELDS = {"run", "task_id", "study_id", "pair_id", "arm", "exclude_reason"}


def read_json(path):
    """Read artifact metadata without importing or executing candidate code."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def digest(path):
    """Hash artifact bytes in bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_overrides(path):
    """Load optional human-specified identities; never override measured scores."""
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "run" not in reader.fieldnames or set(reader.fieldnames) - OVERRIDE_FIELDS:
            raise ValueError("Manifest columns: run,task_id,study_id,pair_id,arm,exclude_reason")
        rows = {}
        for row in reader:
            name = row.get("run", "").strip()
            if not name or name in rows or None in row:
                raise ValueError("Manifest requires unique, nonempty run directory names")
            rows[name] = {key: value.strip() for key, value in row.items() if value and key != "run"}
        return rows


def identify_run(path, metadata, override):
    """Prefer explicit identities, then infer the existing branch/Pod naming scheme."""
    tag = str(metadata.get("run_tag") or path.name)
    stem = re.sub(r"-" + UUID + r"$", "", tag)
    arm = override.get("arm") or metadata.get("experiment_arm") or metadata.get("arm", "")
    if not arm:
        match = re.search(r"-(baseline|analogy)$", stem)
        arm = match.group(1) if match else ""
    pair = override.get("pair_id") or metadata.get("pair_id", "")
    if not pair and arm and stem.endswith("-" + arm):
        pair = stem[:-len(arm) - 1]
    task = override.get("task_id") or metadata.get("task_id", "")
    if not task and arm:
        match = re.fullmatch(r"run/\d{8}_\d{6}-(.+)-" + re.escape(arm), metadata.get("branch", ""))
        task = match.group(1) if match else ""
    task_hash = metadata.get("task_file_sha256", "")
    task = task or ("task-" + task_hash[:12] if task_hash else "")
    study = override.get("study_id") or metadata.get("study_id") or re.sub(r"-pair-\d+$", "", pair)
    return {"run": path.name, "run_id": tag, "path": str(path), "task_id": task,
            "task_hash": task_hash, "study_id": study, "pair_id": pair, "arm": arm,
            "seed": metadata.get("initial_seed", metadata.get("seed", "")),
            "gpu": metadata.get("gpu", ""), "start_commit": metadata.get("start_commit", metadata.get("base_commit", "")),
            "prepared_id": metadata.get("prepared_id", ""),
            "resources": metadata.get("resources", ""),
            "budget_seconds": metadata.get("budget_seconds", ""),
            "agent_model": metadata.get("agent_model", "")}


def read_trial(path, run):
    """Require a completed, hash-bound result with a finite primary metric."""
    receipt = read_json(path / "source.json")
    if receipt.get("execution_status") != "completed":
        raise ValueError("not_completed")
    if receipt.get("protocol") != "autoresearch-experiment-v1":
        raise ValueError("unknown_receipt_protocol")
    if receipt.get("run_id") != run["run_id"] or receipt.get("experiment_id") != path.name:
        raise ValueError("receipt_identity_mismatch")
    bindings = {"source.py": "sha256", "metrics.json": "metrics_sha256", "config.json": "config_sha256"}
    if "log_sha256" in receipt:
        bindings["run.log"] = "log_sha256"
    for name, field in bindings.items():
        if digest(path / name) != receipt.get(field):
            raise ValueError("hash_mismatch:" + name)
    if not (path / "validation_predictions.npy").is_file():
        raise ValueError("missing_validation_predictions")
    metrics = read_json(path / "metrics.json")
    if not run["prepared_id"] or metrics.get("prepared_id") != run["prepared_id"] or receipt.get("prepared_id") != run["prepared_id"]:
        raise ValueError("prepared_id_mismatch")
    score = metrics.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("missing_or_nonfinite_score")
    if not isinstance(metrics.get("maximize"), bool) or not metrics.get("metric_version"):
        raise ValueError("missing_metric_identity_or_direction")
    return {"trial": path.name, "score": score, "metric_version": metrics["metric_version"],
            "maximize": metrics["maximize"], "validation_rows": metrics.get("validation_rows", ""),
            "source_sha256": receipt["sha256"]}


def inventory_runs(root, overrides=None):
    """Keep every run in the inventory, excluding empty/invalid results from charts."""
    overrides = overrides or {}
    paths = [root] if (root / "run.json").is_file() else sorted(p.parent for p in root.glob("*/run.json"))
    unknown = set(overrides) - {p.name for p in paths}
    if unknown:
        raise ValueError("Manifest references missing runs: " + ", ".join(sorted(unknown)))
    records, trials = [], []
    for path in paths:
        row = {"run": path.name, "path": str(path), "status": "excluded", "reason": "",
               "warnings": "", "n_trials": 0, "n_valid_trials": 0}
        try:
            override = overrides.get(path.name, {})
            row.update(identify_run(path, read_json(path / "run.json"), override))
            candidates = []
            for trial in sorted((path / "trials").glob("*")):
                if not trial.is_dir():
                    continue
                row["n_trials"] += 1
                trial_row = {"run": path.name, "trial": trial.name, "status": "excluded", "reason": ""}
                try:
                    candidate = read_trial(trial, row)
                    candidates.append(candidate)
                    trial_row.update(candidate, status="eligible")
                except (OSError, ValueError, TypeError) as exc:
                    trial_row["reason"] = str(exc)
                trials.append(trial_row)
            row["n_valid_trials"] = len(candidates)
            if not candidates:
                row["reason"] = "no_valid_completed_result"
            elif len({(c["metric_version"], c["maximize"], c["validation_rows"]) for c in candidates}) != 1:
                row["reason"] = "inconsistent_trial_metrics"
            else:
                best = sorted(candidates, key=lambda c: (-c["score"] if c["maximize"] else c["score"], c["trial"]))[0]
                row.update(best, status="eligible")
                notes = []
                for name, field in (("prepare.py", "evaluator_sha256"), ("uv.lock", "environment_sha256")):
                    snapshot = path / "_worktree" / name
                    row[field] = digest(snapshot) if snapshot.is_file() else ""
                if not (path / "summary.md").is_file():
                    notes.append("run_completion_not_documented")
                # best.json is a pointer/check only; rounded scores there never replace metrics.json.
                if (path / "best.json").is_file():
                    try:
                        pointer = read_json(path / "best.json")
                        if pointer.get("trial_id") != best["trial"] or pointer.get("source_sha256") != best["source_sha256"]:
                            notes.append("best_pointer_differs_from_verified_optimum")
                    except (OSError, ValueError):
                        notes.append("unreadable_best_pointer")
                row["warnings"] = ";".join(notes)
            if override.get("exclude_reason"):
                row.update(status="excluded", reason="manual:" + override["exclude_reason"])
        except (OSError, ValueError, TypeError) as exc:
            row["reason"] = "invalid_run_metadata:" + str(exc)
        records.append(row)
    return records, trials


def build_pairs(records, reference="baseline", contrasts=None):
    """Pair unique attempts within a draw; never borrow or cherry-pick repeated runs."""
    groups = defaultdict(list)
    pairs, issues = [], []
    for run in records:
        if run["status"] != "eligible":
            continue
        if not all(run.get(key) for key in ("task_id", "study_id", "pair_id", "arm")):
            issues.append({"run": run["run"], "reason": "missing_identity_use_manifest"})
            continue
        groups[(run["study_id"], run["task_id"], run["pair_id"])].append(run)
    for (study, task, pair_id), runs in sorted(groups.items()):
        by_arm = defaultdict(list)
        for run in runs:
            by_arm[run["arm"]].append(run)
        comparisons = contrasts if contrasts is not None else [(arm, reference) for arm in sorted(by_arm) if arm != reference]
        if not comparisons:
            issues.append({"study_id": study, "task_id": task, "pair_id": pair_id, "reason": "no_comparison_arm"})
        for treatment, control in comparisons:
            identity = {"study_id": study, "task_id": task, "pair_id": pair_id,
                        "treatment": treatment, "reference": control}
            if len(by_arm[treatment]) != 1 or len(by_arm[control]) != 1:
                issues.append({**identity, "reason": "missing_or_ambiguous_arm", "runs": ";".join(r["run"] for r in runs)})
                continue
            a, b = by_arm[treatment][0], by_arm[control][0]
            mismatches = [field for field in ("prepared_id", "metric_version", "maximize", "validation_rows", "task_hash", "evaluator_sha256")
                          if a.get(field) != b.get(field)]
            if mismatches:
                issues.append({**identity, "reason": "incompatible:" + ",".join(mismatches)})
                continue
            warnings = {note for r in (a, b) for note in r["warnings"].split(";") if note}
            for field in ("gpu", "start_commit", "seed", "resources", "environment_sha256", "budget_seconds", "agent_model"):
                if a.get(field) in (None, "") or b.get(field) in (None, ""):
                    warnings.add(field + "_unknown")
                elif a[field] != b[field]:
                    warnings.add(field + "_differs")
            for field in ("task_hash", "evaluator_sha256", "validation_rows"):
                if a.get(field) in (None, ""):
                    warnings.add(field + "_unknown")
            cohort = [study, task, a["metric_version"], a["maximize"], a["task_hash"], a["evaluator_sha256"]]
            cohort_id = hashlib.sha256(json.dumps(cohort).encode()).hexdigest()[:10]
            delta = a["score"] - b["score"]
            pairs.append({**identity, "cohort": cohort_id, "metric_version": a["metric_version"],
                          "maximize": a["maximize"], "prepared_id": a["prepared_id"],
                          "treatment_run": a["run"], "reference_run": b["run"],
                          "treatment_score": a["score"], "reference_score": b["score"],
                          "effect": delta if a["maximize"] else -delta,
                          "warnings": ";".join(sorted(warnings))})
    return pairs, issues


def summarize_effects(pairs):
    """Compute paired mean differences and t intervals across independent draws."""
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["cohort"], pair["treatment"], pair["reference"])].append(pair)
    summaries = []
    for (_, treatment, reference), rows in sorted(groups.items()):
        values = [row["effect"] for row in rows]
        mean, n = statistics.mean(values), len(values)
        low = high = None
        if n > 1:
            from scipy.stats import t
            half = float(t.ppf(0.975, n - 1)) * statistics.stdev(values) / math.sqrt(n)
            low, high = mean - half, mean + half
        summaries.append({key: rows[0][key] for key in ("cohort", "study_id", "task_id", "metric_version", "maximize")} |
                         {"treatment": treatment, "reference": reference, "n_pairs": n,
                          "mean_effect": mean, "ci95_low": low, "ci95_high": high,
                          "flagged_pairs": sum(bool(row["warnings"]) for row in rows)})
    return summaries


def write_csv(path, rows):
    """Write auditable tables, including a header when no records qualify."""
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["reason"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_figures(pairs, effects, records, output):
    """Draw matched run scores and signed differences, separately for each cohort."""
    if not pairs:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#172033", "axes.edgecolor": "#cbd5e1",
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    cohorts = defaultdict(list)
    for pair in pairs:
        cohorts[pair["cohort"]].append(pair)
    generated = []
    for cohort, rows in sorted(cohorts.items()):
        first = rows[0]
        title = "\n".join(textwrap.wrap(first["task_id"].replace("-", " "), 80))
        subtitle = f"Study: {first['study_id']}  |  {first['metric_version']}"
        excluded = sum(r["status"] != "eligible" for r in records)
        warnings = sorted({w for row in rows for w in row["warnings"].split(";") if w})
        flagged = len({row["pair_id"] for row in rows if row["warnings"]})
        footnote = f"Best verified completed validation result per run; {excluded}/{len(records)} input runs excluded."
        if warnings:
            footnote += f"  {flagged} pair(s) flagged: " + ", ".join(w.replace("_", " ") for w in warnings) + "."
        footnote += " Descriptive comparison; trials are not independent repeats."
        footnote = textwrap.fill(footnote, 135)
        footer_height = 0.11 + 0.022 * footnote.count("\n")
        stem = re.sub(r"[^\w.-]+", "-", first["task_id"] + "__" + first["study_id"])[:100] + "__" + cohort

        def save(fig, kind):
            fig.text(0.07, 0.035, footnote, fontsize=8, color="#64748b", va="bottom", linespacing=1.6)
            for extension in ("png", "pdf"):
                path = charts / f"{stem}_{kind}.{extension}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                generated.append(path.relative_to(output).as_posix())
            plt.close(fig)

        draws = defaultdict(dict)
        draw_flags = defaultdict(bool)
        for row in rows:
            draws[row["pair_id"]][row["reference"]] = row["reference_score"]
            draws[row["pair_id"]][row["treatment"]] = row["treatment_score"]
            draw_flags[row["pair_id"]] |= bool(row["warnings"])
        references = {row["reference"] for row in rows}
        all_arms = {arm for draw in draws.values() for arm in draw}
        arms = sorted(all_arms, key=lambda a: (a not in references, a))
        fig, ax = plt.subplots(figsize=(11.8, max(6.3, 3.2 + 0.22 * len(draws))))
        fig.subplots_adjust(left=0.10, right=0.74, top=0.76, bottom=footer_height + 0.08)
        fig.suptitle(title, x=0.10, y=0.96, ha="left", fontsize=15, fontweight="bold")
        fig.text(0.10, 0.865, subtitle + "\nPaired scores · one line per experimental pair", fontsize=10, color="#64748b")
        for index, (pair_id, scores) in enumerate(sorted(draws.items())):
            xs = [i for i, arm in enumerate(arms) if arm in scores]
            ys = [scores[arms[i]] for i in xs]
            color = plt.get_cmap("tab10")(index % 10)
            label = textwrap.fill(pair_id, 26) + (" *" if draw_flags[pair_id] else "")
            ax.plot(xs, ys, marker="o", markersize=7, linewidth=1.8, alpha=0.85,
                    linestyle="--" if draw_flags[pair_id] else "-", color=color, label=label)
            if len(draws) <= 8:
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:.6f}", (x, y), xytext=(0, 10), textcoords="offset points",
                                ha="center", fontsize=9, color=color)
        ax.set_xticks(range(len(arms)), [textwrap.fill(a, 20) for a in arms])
        ax.set_xlim(-0.30, len(arms) - 0.70)
        ax.margins(y=0.25)
        if not first["maximize"]:
            ax.invert_yaxis()
        ax.set_ylabel("Best validation score (" + ("higher" if first["maximize"] else "lower") + " is better)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1), frameon=False, fontsize=9,
                  title=f"Pairs: {len(draws)}" + ("  (* flagged)" if flagged else ""))
        save(fig, "paired")

        stats = [s for s in effects if s["cohort"] == cohort]
        fig, ax = plt.subplots(figsize=(11.8, max(5.8, 3.8 + 0.85 * len(stats))))
        fig.subplots_adjust(left=0.28, right=0.94, top=0.72, bottom=footer_height + 0.11)
        fig.suptitle(title, x=0.07, y=0.96, ha="left", fontsize=15, fontweight="bold")
        interval_note = "95% t intervals require at least 2 pairs; none shown here." if all(s["n_pairs"] == 1 for s in stats) else "Dots: experimental pairs. Diamond/bar: mean and 95% paired t interval (n ≥ 2)."
        fig.text(0.07, 0.84, subtitle + "\nPaired effects · " + interval_note, fontsize=10, color="#64748b")
        bounds = [0.0]
        for index, stat in enumerate(stats):
            matched = [r for r in rows if (r["treatment"], r["reference"]) == (stat["treatment"], stat["reference"])]
            values = [r["effect"] for r in matched]
            bounds.extend(values)
            jitter = [(i - (len(values) - 1) / 2) * min(0.06, 0.3 / len(values)) for i in range(len(values))]
            ax.scatter(values, [index + j for j in jitter], s=55, color="#2563eb", alpha=0.8, zorder=3)
            if stat["n_pairs"] > 1:
                bounds.extend([stat["ci95_low"], stat["ci95_high"]])
                ax.errorbar(stat["mean_effect"], index + 0.25,
                            xerr=[[stat["mean_effect"] - stat["ci95_low"]], [stat["ci95_high"] - stat["mean_effect"]]],
                            fmt="D", markersize=5, color="#172033", capsize=4, linewidth=1.7)
            ax.annotate(f"mean {stat['mean_effect']:+.6f}", (stat["mean_effect"], index),
                        xytext=(0, 15), textcoords="offset points", ha="center", fontsize=9)
        span = max(bounds) - min(bounds) or max(abs(bounds[0]), 1.0) * 0.02
        ax.set_xlim(min(bounds) - span * 0.25, max(bounds) + span * 0.25)
        ax.set_yticks(range(len(stats)), [f"{s['treatment']} vs {s['reference']}\n(n={s['n_pairs']} pairs)" for s in stats])
        ax.set_ylim(len(stats) - 0.35, -0.65)
        ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2)
        ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlabel("Signed improvement in metric units\nPositive = comparison arm better; negative = reference better")
        save(fig, "effect")
    return generated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path.home() / "nautilus" / "autoresearch-result")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "analysis")
    parser.add_argument("--manifest", type=Path, help="CSV identity overrides or explicit exclusions")
    parser.add_argument("--reference", default="baseline", help="default reference arm")
    parser.add_argument("--contrast", action="append", help="explicit treatment:reference; repeat for multiple contrasts")
    args = parser.parse_args(argv)
    args.runs, args.out = args.runs.expanduser().resolve(), args.out.expanduser().resolve()
    if not args.runs.is_dir():
        parser.error("--runs must be an existing result directory")
    contrasts = None
    if args.contrast:
        contrasts = []
        for value in args.contrast:
            parts = value.split(":")
            if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
                parser.error("--contrast requires two different arms: treatment:reference")
            if tuple(parts) not in contrasts:
                contrasts.append(tuple(parts))
    records, trials = inventory_runs(args.runs, load_overrides(args.manifest))
    pairs, issues = build_pairs(records, args.reference, contrasts)
    effects = summarize_effects(pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    previous = args.out / "analysis.json"
    old_charts = read_json(previous).get("charts", []) if previous.exists() else []
    charts = plot_figures(pairs, effects, records, args.out)
    for name, rows in (("run_inventory", records), ("trial_inventory", trials),
                       ("pairs", pairs), ("effects", effects), ("pair_issues", issues)):
        write_csv(args.out / (name + ".csv"), rows)
    # Retire only charts recorded by this script, so filtering cannot leave stale figures.
    for relative in set(old_charts) - set(charts):
        path = args.out / relative
        if path.parent == args.out / "charts" and path.suffix in (".png", ".pdf"):
            path.unlink(missing_ok=True)
    eligible = sum(r["status"] == "eligible" for r in records)
    summary = ["# Autoresearch result analysis", "",
               f"Input: `{args.runs}`", f"Runs: {len(records)}; eligible: {eligible}; excluded: {len(records) - eligible}.",
               f"Matched contrasts: {len(pairs)}; pairing issues: {len(issues)}.", "",
               "Scores are best verified completed **validation** results, not held-out test scores. "
               "Each independent experimental pair counts once per contrast; individual trials are not repeats. "
               "Effects are direction-corrected (positive means the comparison arm is better). "
               "95% t intervals use variation across pairs and assume independent draws; n=1 has no interval. "
               "Different studies, tasks, metric versions, task hashes and evaluator hashes are never pooled. "
               "Different prepared splits may enter one cohort only through separate internally matched pairs.", ""]
    for stat in effects:
        interval = "unavailable (n=1)" if stat["ci95_low"] is None else f"[{stat['ci95_low']:+.6f}, {stat['ci95_high']:+.6f}]"
        summary.append(f"- {stat['study_id']} / {stat['task_id']} / {stat['treatment']} vs {stat['reference']}: "
                       f"mean effect **{stat['mean_effect']:+.6f}**, n={stat['n_pairs']}, 95% CI {interval}.")
    summary.extend(["", "## Comparability notes", ""])
    summary.extend(f"- {p['pair_id']} ({p['treatment']} vs {p['reference']}): {p['warnings'] or 'no recorded mismatch'}." for p in pairs)
    summary.extend(["", "GPU, initial commit, model and budget differences/unknowns are flags, not automatic exclusions. "
                    "A flagged pair describes observed outcomes and does not isolate the treatment's causal effect. "
                    "A completed trial does not prove that the entire run exhausted its intended budget. "
                    "No timestamp proximity or shared training seed is used to invent pairings. "
                    "Consult run_inventory.csv for identities/hardware, trial_inventory.csv for rejected candidates, "
                    "and pair_issues.csv for ambiguous or incompatible pairs.", ""])
    (args.out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    previous.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "script_sha256": digest(Path(__file__)), "runs": str(args.runs),
                                   "manifest": str(args.manifest) if args.manifest else None,
                                   "reference": args.reference, "contrasts": contrasts,
                                   "n_runs": len(records), "n_eligible": eligible, "charts": charts}, indent=2) + "\n")
    print(f"Runs: {len(records)}; eligible: {eligible}; excluded: {len(records) - eligible}; matched contrasts: {len(pairs)}")
    for stat in effects:
        print(f"  {stat['study_id']}: {stat['treatment']} vs {stat['reference']}, signed effect = {stat['mean_effect']:+.6f} (n={stat['n_pairs']})")
    print(f"Output: {args.out}")
    if not pairs:
        print("No unambiguous compatible pairs; audit tables explain exclusions. No charts generated.")
    return 0 if pairs else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit(f"Analysis failed: {exc}\nRun with: uv run --script analyze_runs.py") from exc
