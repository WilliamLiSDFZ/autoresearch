"""Evidence-aware report validation and atomic rendering for context version 2."""
from __future__ import annotations

import copy
import json
import math
import re


ANCHOR_SCHEMA = {"type": "object", "properties": {
    "node_id": {"type": "string"}, "source_sha256": {"type": "string"},
    "start_line": {"type": "integer", "minimum": 1},
    "end_line": {"type": "integer", "minimum": 1}},
    "required": ["node_id", "source_sha256", "start_line", "end_line"],
    "additionalProperties": False}


# 扩展报告工具结构，加入事实、假设及机制证据字段。
def extend_tools(tools, *, mode):
    tools = copy.deepcopy(tools)
    report = next(t["function"]["parameters"] for t in tools
                  if t["function"]["name"] == "submit_report")
    report["properties"].update({
        "report_schema_revision": {"type": "integer", "enum": [2],
            "description": "Use 2 for the explicit runtime evidence contract; independent of context_version."},
        "abstention_reason": {"type": "string", "description":
            "Explain why no grounded mechanism is suitable when mechanisms is empty; otherwise omit."},
        "observed_facts": {"type": "array", "maxItems": 10, "items": {
            "type": "object", "properties": {
                "statement": {"type": "string"}, "evidence": {"type": "string",
                    "description": "Human-readable evidence explanation; put runtime paths in runtime_evidence."},
                "runtime_evidence": {"type": "array", "items": {"type": "string"},
                    "description": "For runtime facts: all supporting paths relative to the visible runtime_context, "
                    "one path per item, numeric list indices as .0. Empty for code/task facts."},
                "source": {"type": "string", "enum": ["code", "runtime", "task"]},
                "code_refs": {"type": "array", "items": ANCHOR_SCHEMA}},
            "required": ["statement", "evidence", "source"]}},
        "hypotheses": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "unknowns": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    })
    report["required"] += ["observed_facts", "hypotheses", "unknowns"]
    mechanism = report["properties"]["mechanisms"]["items"]
    mechanism["properties"].update({
        "implementation_basis": {"type": "string", "enum": ["code", "runtime", "task"]},
        "code_refs": {"type": "array", "items": ANCHOR_SCHEMA, "maxItems": 6},
        "runtime_evidence": {"type": "array", "items": {"type": "string"},
                             "description": "Exact dotted paths relative to runtime_context in the packet"},
        "history_comparison": {"type": "object", "properties": {
            "related_trial_ids": {"type": "array", "items": {"type": "string"},
                "description": "Related IDs actually visible in experiment_history, including unfinished attempts."},
            "difference": {"type": "string", "minLength": 1,
                "description": "Substantive difference from related attempts, or why none are related."},
            "retry_reason": {"type": "string", "description":
                "Why revisiting related attempts is warranted; may be empty only when no trial is related."}},
            "required": ["related_trial_ids", "difference", "retry_reason"],
            "additionalProperties": False,
            "description": "Required for improve when experiment history is visible; optional without history."},
        **{name: {"type": "string"} for name in
           ("assumptions", "target_fit", "constraints", "validation_plan", "rejection_criterion")},
    })
    mechanism["required"] = list(dict.fromkeys(mechanism["required"] + [
        "implementation_basis", "code_refs", "runtime_evidence", "assumptions", "target_fit",
        "constraints", "validation_plan", "rejection_criterion"]))
    return tools


def _strings(value):
    return isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)


def _history_issues(candidate, location, trial_ids, required):
    at = location + ".history_comparison"
    if "history_comparison" not in candidate and not required:
        return []
    comparison = candidate.get("history_comparison")
    if not isinstance(comparison, dict):
        return [_issue("history_comparison_required", at, comparison,
            "a history comparison object", "Compare this mechanism with the visible experiment history.")]
    issues = []
    related = comparison.get("related_trial_ids")
    if not _strings(related):
        issues.append(_issue("history_trial_ids_type", at + ".related_trial_ids", related,
            "an array of nonempty trial IDs", "Use [] when no visible attempt is related."))
    else:
        for index, trial_id in enumerate(related):
            if trial_id not in trial_ids:
                issues.append(_issue("history_trial_unavailable", f"{at}.related_trial_ids[{index}]", trial_id,
                    "a trial ID in the visible experiment history", "Copy an ID actually shown in this episode."))
    difference = comparison.get("difference")
    if not isinstance(difference, str) or not difference.strip():
        issues.append(_issue("history_difference_required", at + ".difference", difference,
            "a nonempty comparison", "Explain the substantive change or why no previous attempt is related."))
    reason = comparison.get("retry_reason")
    if not isinstance(reason, str) or (isinstance(related, list) and related and not reason.strip()):
        issues.append(_issue("history_retry_reason_required", at + ".retry_reason", reason,
            "a string, nonempty when related trials are listed",
            "Explain why revisiting these attempts is warranted; distinguish incomplete runs from measured failures."))
    return issues


def _runtime_path(path, data):
    if not isinstance(path, str) or not path:
        return False
    value = data
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return False
    return (value is not None and (not isinstance(value, str)
            or bool(value.strip()) and value.strip().lower() != "unknown")
            and (not isinstance(value, float) or math.isfinite(value)))


def _issue(code, location, received, expected, repair_hint):
    return {"code": code, "location": location, "received": copy.deepcopy(received),
            "expected": expected, "repair_hint": repair_hint}


# 将结构化校验问题转换为兼容的可读说明。
def format_issue(issue):
    """Compatibility text; new callers should retain the structured issue too."""
    return f"{issue['location']}: {issue['code']}: {issue['expected']}. {issue['repair_hint']}"


def _normalize_paths(raw, location, normalizations):
    # Compatibility is deliberately syntactic: no guessing keys, dropping references,
    # substituting values or stripping a runtime_context prefix.
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        return raw
    normalized = [re.sub(r"\[(0|[1-9][0-9]*)\]", r".\1", part.strip())
                  for value in values for part in value.split(";")]
    if normalized != raw:
        normalizations.append({"location": location, "received": copy.deepcopy(raw),
                               "normalized": copy.deepcopy(normalized)})
    return normalized


def _runtime_issues(refs, location, data):
    from .context import runtime_evidence_paths
    if not _strings(refs):
        return [_issue("runtime_evidence_type", location, refs, "an array of nonempty runtime paths",
                       "Use one path per array item; numeric list indices use .0.")]
    issues = []
    for index, path in enumerate(refs):
        if not _runtime_path(path, data):
            root = path.split(".", 1)[0]
            visible = runtime_evidence_paths(data)
            nearby = [p for p in visible if p.split(".", 1)[0] == root] or visible
            issues.append(_issue("runtime_path_unavailable", f"{location}[{index}]", path,
                "a path to a non-null, non-unknown value actually shown in this episode",
                "Copy a visible path, read evidence where supported, or revise the claim. "
                "Visible path examples: " + json.dumps(nearby[:8], ensure_ascii=False)))
    return issues


def _anchor_issues(refs, location, session):
    if not isinstance(refs, list) or len(refs) > 6:
        return [_issue("code_refs_type", location, refs, "an array of at most six source anchors",
                       "Use exact node_id, source_sha256, start_line and end_line from source tools.")]
    issues = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict) or session is None or not session.validate_anchor(ref):
            available = [] if session is None else getattr(session, "anchors", [])
            if isinstance(ref, dict):
                available = [a for a in available if a.get("node_id") == ref.get("node_id")]
            issues.append(_issue("code_anchor_unread", f"{location}[{index}]", ref,
                "exact node/hash and lines actually returned by source tools or the packet index",
                "Read missing lines or narrow/remove the claim; available read anchors: "
                + json.dumps(available[:8], ensure_ascii=False)))
    return issues


def _paper_issues(candidate, location, seen_ids, corpus, reading, abstracts):
    issues = []
    ids = candidate.get("paper_ids")
    if not _strings(ids) or not ids or len(ids) > 4:
        issues.append(_issue("paper_ids_invalid", location + ".paper_ids", ids,
            "one to four paper IDs returned by search in this episode",
            "Cite the exact returned paper IDs."))
        ids = ids if isinstance(ids, list) else []
    for index, pid in enumerate(ids):
        if not isinstance(pid, str) or pid not in seen_ids or pid not in corpus:
            issues.append(_issue("paper_not_seen", f"{location}.paper_ids[{index}]", pid,
                "a paper ID returned by search and present in the corpus",
                "Search for this paper or remove the unsupported mechanism."))
    if reading is None:
        return issues
    refs = candidate.get("evidence_refs")
    if not isinstance(refs, list) or not refs or len(refs) > 6:
        issues.append(_issue("paper_evidence_required", location + ".evidence_refs", refs,
            "one to six references to text actually read this episode",
            "Use read_abstract/read_paper, then copy an exact 12–400 character quote."))
        return issues
    supported = set()
    for index, ref in enumerate(refs):
        at = f"{location}.evidence_refs[{index}]"
        if not isinstance(ref, dict):
            issues.append(_issue("paper_evidence_type", at, ref, "a paper evidence object",
                                 "Use paper_id, source, quote and chunk_id for full_text."))
            continue
        pid, source, quote = ref.get("paper_id"), ref.get("source"), ref.get("quote")
        if not isinstance(pid, str) or pid not in ids:
            issues.append(_issue("paper_evidence_id", at + ".paper_id", pid,
                "a paper ID in this mechanism's paper_ids", "Use the exact cited paper ID."))
            continue
        if not isinstance(quote, str) or not 12 <= len(quote.strip()) <= 400:
            issues.append(_issue("paper_quote_length", at + ".quote", quote,
                "an exact quote of 12–400 characters", "Copy a short contiguous quote from returned text."))
            continue
        if source == "full_text":
            chunk_id = ref.get("chunk_id")
            chunk = reading.delivered.get((pid, chunk_id)) if isinstance(chunk_id, str) else None
            text = chunk.get("text", "") if chunk else ""
        elif source == "abstract":
            text = abstracts.get(pid, "")
        else:
            text = ""
        if not text:
            issues.append(_issue("paper_source_not_returned", at, ref,
                "an abstract or full-text chunk actually returned in this episode",
                "Read this source or remove the unsupported mechanism. "
                "Opening a paper alone does not count as reading its body."))
        else:
            # 已读来源是硬约束；逐字匹配由下层标记警告，不阻断交付。
            supported.add(pid)
    for index, pid in enumerate(ids):
        if isinstance(pid, str) and pid not in supported:
            issues.append(_issue("paper_read_evidence_missing", f"{location}.paper_ids[{index}]", pid,
                "at least one valid read reference for each cited paper",
                "Read the paper and cite returned text, or remove the unsupported mechanism."))
    return issues


# 校验报告结构与证据引用，分别保留合格机制和拒绝原因。
def validate_detailed(report, seen_ids, corpus, max_mechanisms, *, reading, abstracts,
                      code_session, runtime_context, mode):
    """Validate shared facts and whole mechanisms; retain quote mismatches as warnings.

    ``normalized_report`` preserves every submitted item, including rejected ones.
    ``report`` contains only independently valid complete mechanisms. Invalid shared
    facts block all delivery because their mechanism dependencies are unknown.
    ``mechanism_mapping`` uses zero-based original indices; final IDs are assigned
    after validation and must be filtered again if rendering removes whole blocks.
    """
    from .agent import validate_report
    empty = {"context_version": 2, "report_schema_revision": 2, "bottlenecks": [], "mechanisms": [],
             "observed_facts": [], "hypotheses": [], "unknowns": []}
    result = {"report": copy.deepcopy(empty), "normalized_report": copy.deepcopy(report), "issues": [],
              "mechanism_mapping": [], "dropped_mechanisms": [], "normalizations": [],
              "shared_facts_valid": False}
    issues = result["issues"]
    if not isinstance(report, dict):
        issues.append(_issue("report_type", "$", report, "a report object", "Submit an object using the tool schema."))
        return result
    raw = result["normalized_report"]
    revision = raw.get("report_schema_revision")
    if revision is not None and (type(revision) is not int or revision != 2):
        issues.append(_issue("report_schema_revision", "report_schema_revision", revision,
            "2 (or an omitted marker for a legacy report)", "Use report_schema_revision=2."))
    raw["report_schema_revision"] = 2
    facts = raw.get("observed_facts")
    if not isinstance(facts, list) or len(facts) > 10:
        issues.append(_issue("observed_facts_type", "observed_facts", facts,
            "an array of at most ten facts", "Use [] when no fact is confirmed."))
        facts = []
    for field in ("hypotheses", "unknowns"):
        if not _strings(raw.get(field)) or len(raw[field]) > 10:
            issues.append(_issue("report_array_type", field, raw.get(field),
                "an array of at most ten nonempty strings", "Use [] when there are no entries."))
    for index, fact in enumerate(facts):
        at = f"observed_facts[{index}]"
        if not isinstance(fact, dict):
            issues.append(_issue("observed_fact_type", at, fact, "a fact object", "Use statement, evidence and source."))
            continue
        for field in ("statement", "evidence", "source"):
            if not isinstance(fact.get(field), str) or not fact[field].strip():
                issues.append(_issue("observed_fact_field", at + "." + field, fact.get(field),
                    "a nonempty string", "Supply a concrete fact and its evidence explanation."))
        source = fact.get("source")
        if not isinstance(source, str) or source not in {"code", "runtime", "task"}:
            issues.append(_issue("observed_fact_source", at + ".source", source,
                "code, runtime or task", "Use the source that actually supports this fact."))
        refs = fact.get("code_refs", [])
        issues.extend(_anchor_issues(refs, at + ".code_refs", code_session))
        if source == "code" and not refs:
            issues.append(_issue("code_evidence_required", at + ".code_refs", refs,
                "at least one actually read code anchor", "Read the relevant code lines and copy their anchor."))
        paths = fact.get("runtime_evidence", [])
        if revision == 2 and "runtime_evidence" in fact and not isinstance(paths, list):
            issues.append(_issue("runtime_evidence_type", at + ".runtime_evidence", paths,
                "an array of runtime paths", "For revision 2, use one path per array item."))
        if source == "runtime" and "runtime_evidence" not in fact:
            if revision == 2:
                issues.append(_issue("runtime_evidence_required", at + ".runtime_evidence", None,
                    "a nonempty array of visible runtime paths", "Move paths from evidence into runtime_evidence."))
            else:
                paths = fact.get("evidence")
        paths = _normalize_paths(paths, at + ".runtime_evidence", result["normalizations"])
        fact["runtime_evidence"] = paths
        issues.extend(_runtime_issues(paths, at + ".runtime_evidence", runtime_context))
        if source == "runtime" and paths == []:
            issues.append(_issue("runtime_evidence_required", at + ".runtime_evidence", paths,
                "at least one visible runtime path", "Copy a supporting path from the visible runtime_context."))
    mechanisms = raw.get("mechanisms")
    if not isinstance(mechanisms, list):
        issues.append(_issue("mechanisms_type", "mechanisms", mechanisms, "an array",
                             "Use [] and abstention_reason if no grounded mechanism is suitable."))
        mechanisms = []
    bottlenecks = raw.get("bottlenecks")
    if not isinstance(bottlenecks, list) or any(not isinstance(b, dict) or
            not isinstance(b.get("statement"), str) or not b["statement"].strip() for b in bottlenecks):
        issues.append(_issue("bottlenecks_type", "bottlenecks", bottlenecks,
            "an array of bottleneck objects with nonempty statements", "Keep indices stable while correcting bottlenecks."))
    if mechanisms and not bottlenecks:
        issues.append(_issue("bottleneck_required", "bottlenecks", bottlenecks,
            "at least one bottleneck addressed by the mechanisms", "State the diagnosed bottleneck first."))
    if raw.get("abstention_reason") is not None and not isinstance(raw["abstention_reason"], str):
        issues.append(_issue("abstention_reason_type", "abstention_reason", raw["abstention_reason"],
            "a string", "Explain the decision using text, or omit this field when proposing mechanisms."))
    if not mechanisms and revision == 2 and (not isinstance(raw.get("abstention_reason"), str)
                                           or not raw["abstention_reason"].strip()):
        issues.append(_issue("abstention_reason_required", "abstention_reason", raw.get("abstention_reason"),
            "a reason for submitting no mechanisms", "Explain why no supported mechanism is suitable."))
    if issues:
        result["dropped_mechanisms"] = [{"original_index": i, "issues": copy.deepcopy(issues)}
                                        for i in range(len(mechanisms))]
        return result
    result["shared_facts_valid"] = True
    clean = {**empty, "observed_facts": copy.deepcopy(facts), "hypotheses": list(raw["hypotheses"]),
             "unknowns": list(raw["unknowns"])}
    if raw.get("abstention_reason"):
        clean["abstention_reason"] = raw["abstention_reason"]
    result["report"] = clean
    history = runtime_context.get("experiment_history", []) if isinstance(runtime_context, dict) else []
    trial_ids = {record["trial_id"] for record in history
                 if isinstance(record, dict) and isinstance(record.get("trial_id"), str) and record["trial_id"].strip()
                 } if isinstance(history, list) else set()
    fields = ("assumptions", "target_fit", "constraints", "validation_plan", "rejection_criterion")
    for index, candidate in enumerate(mechanisms):
        at = f"mechanisms[{index}]"
        errors = []
        if not isinstance(candidate, dict):
            errors.append(_issue("mechanism_type", at, candidate, "a complete mechanism object",
                                 "Use the submit_report mechanism schema."))
        else:
            errors.extend(_history_issues(candidate, at, trial_ids, mode == "improve" and bool(trial_ids)))
            for field in (*fields, "title", "mechanism", "intervention", *(("limitations",) if reading is not None else ())):
                if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                    errors.append(_issue("mechanism_field_required", at + "." + field, candidate.get(field),
                        "a nonempty string", "Complete this field or remove the whole mechanism."))
            bottleneck_index = candidate.get("bottleneck_idx", 0)
            if type(bottleneck_index) is not int or not 0 <= bottleneck_index < len(bottlenecks):
                errors.append(_issue("bottleneck_index_invalid", at + ".bottleneck_idx", bottleneck_index,
                    "a zero-based index of a submitted bottleneck", "Copy the correct bottleneck index."))
            basis = candidate.get("implementation_basis")
            refs = candidate.get("code_refs", [])
            errors.extend(_anchor_issues(refs, at + ".code_refs", code_session))
            runtime_refs = _normalize_paths(candidate.get("runtime_evidence", []),
                                            at + ".runtime_evidence", result["normalizations"])
            if revision == 2 and not isinstance(candidate.get("runtime_evidence", []), list):
                errors.append(_issue("runtime_evidence_type", at + ".runtime_evidence",
                    candidate.get("runtime_evidence"), "an array of runtime paths",
                    "For revision 2, use one path per array item."))
            candidate["runtime_evidence"] = runtime_refs
            errors.extend(_runtime_issues(runtime_refs, at + ".runtime_evidence", runtime_context))
            if mode == "improve":
                if not isinstance(basis, str) or basis not in {"code", "runtime"}:
                    errors.append(_issue("implementation_basis_invalid", at + ".implementation_basis", basis,
                        "code or runtime for improve", "Ground the intervention in read code or visible runtime evidence."))
                elif (basis == "code" and not refs) or (basis == "runtime" and not runtime_refs):
                    field = "code_refs" if basis == "code" else "runtime_evidence"
                    errors.append(_issue("implementation_evidence_required", at + "." + field, candidate.get(field),
                        "at least one reference supporting the implementation basis", "Read and cite the required evidence."))
            elif basis != "task":
                errors.append(_issue("implementation_basis_invalid", at + ".implementation_basis", basis,
                    "task for draft", "Draft has no executed candidate; use implementation_basis=task."))
            errors.extend(_paper_issues(candidate, at, seen_ids, corpus, reading, abstracts or {}))
        if not errors:
            try:
                legacy, legacy_issues = validate_report({"bottlenecks": bottlenecks, "mechanisms": [candidate]},
                    seen_ids, corpus, 1, reading=reading, abstracts=abstracts)
                for problem in legacy_issues:
                    errors.append(_issue("mechanism_contract", at, candidate,
                                         problem, "Correct the cited fields without weakening the evidence."))
                if not legacy["mechanisms"] and not errors:
                    errors.append(_issue("mechanism_rejected", at, candidate,
                        "a complete, grounded mechanism", "Correct the mechanism using the tool schema."))
            except (TypeError, ValueError, KeyError):
                errors.append(_issue("mechanism_malformed", at, candidate,
                    "fields matching the tool schema", "Correct malformed bottleneck or mechanism fields."))
        if not errors and len(clean["mechanisms"]) >= max_mechanisms:
            errors.append(_issue("mechanism_limit", at, index,
                f"at most {max_mechanisms} mechanisms", "Remove a complete lower-priority mechanism."))
        if errors:
            issues.extend(errors)
            result["dropped_mechanisms"].append({"original_index": index, "issues": errors})
            continue
        item = legacy["mechanisms"][0]
        for key in (*fields, "limitations", "history_comparison"):
            if key in candidate:
                item[key] = copy.deepcopy(candidate[key])
        item.update(implementation_basis=basis, code_refs=copy.deepcopy(refs), runtime_evidence=runtime_refs,
                    mechanism_id=f"m{len(clean['mechanisms']) + 1}")
        clean["bottlenecks"] = legacy["bottlenecks"]
        clean["mechanisms"].append(item)
        result["mechanism_mapping"].append({"original_index": index, "mechanism_id": item["mechanism_id"]})
    if not mechanisms:
        clean["bottlenecks"] = copy.deepcopy(bottlenecks)
    return result


# 以兼容接口返回校验后的报告和可读问题列表。
def validate(report, seen_ids, corpus, max_mechanisms, *, reading, abstracts,
             code_session, runtime_context, mode):
    """Legacy pair contract retained for historical callers and CPU regressions."""
    details = validate_detailed(report, seen_ids, corpus, max_mechanisms, reading=reading,
        abstracts=abstracts, code_session=code_session, runtime_context=runtime_context, mode=mode)
    return details["report"], [format_issue(issue) for issue in details["issues"]]


# 在字符预算内渲染报告，只整块移除超限机制。
def render(report, corpus, budget_chars, mode="improve"):
    """Return the exact report and Markdown retained under the total budget."""
    from .agent import render_report
    clean = copy.deepcopy(report)

    def one_render():
        if not clean["mechanisms"]:
            return ""
        facts = "\n".join(f"- {f['statement']} [source={f['source']}; evidence={f['evidence']}; "
                          f"runtime_evidence={json.dumps(f.get('runtime_evidence', []), ensure_ascii=False)}; "
                          f"code_refs={json.dumps(f.get('code_refs', []), ensure_ascii=False)}]"
                          for f in clean["observed_facts"]) or "- None confirmed."
        prefix = ("## Implementation evidence\n\nObserved facts:\n" + facts + "\n\nHypotheses:\n" +
                  "\n".join(f"- {s}" for s in clean["hypotheses"]) + "\n\nUnknowns:\n" +
                  "\n".join(f"- {s}" for s in clean["unknowns"]) + "\n\n")
        # An effectively unlimited legacy render gives whole mechanisms; this layer
        # applies the one actual budget to the combined evidence and all mechanism fields.
        body = render_report(clean, corpus, 10**12, mode=mode)
        for item in clean["mechanisms"]:
            heading = "### " + item["title"]
            replacement = (heading + f" [{item['mechanism_id']}]\n"
                           f"**Implementation basis**: {item['implementation_basis']}\n"
                           f"**Code locations read**: {json.dumps(item['code_refs'], ensure_ascii=False)}\n"
                           f"**Runtime evidence**: {json.dumps(item['runtime_evidence'])}\n"
                           f"**Constraints to preserve**: {item['constraints']}\n"
                           f"**Explicit rejection condition**: {item['rejection_criterion']}")
            if "history_comparison" in item:
                comparison = item["history_comparison"]
                related = ", ".join(comparison["related_trial_ids"]) or "none"
                replacement += (f"\n**History comparison**: related trials: {related}. "
                                f"{comparison['difference']}")
                if comparison["retry_reason"]:
                    replacement += f" Retry rationale: {comparison['retry_reason']}"
            if "evidence_refs" not in item:
                replacement += (f"\n**Source assumptions**: {item['assumptions']}\n"
                                f"**Fit**: {item['target_fit']}\n**Validation**: {item['validation_plan']}")
            body = body.replace(heading + "\n", replacement + "\n", 1)
        return prefix + body

    while clean["mechanisms"]:
        text = one_render()
        if budget_chars <= 0 or len(text) <= budget_chars:
            return clean, text
        clean["mechanisms"].pop()  # whole lowest-priority mechanisms only
    return clean, ""
