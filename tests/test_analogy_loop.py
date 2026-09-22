"""Offline integration of the actual tool loop, evidence readers and validators."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from autoresearch_analogy.code_tools import CodeReadingSession
from autoresearch_analogy.context import ContextOptions
from autoresearch_analogy.corpus import PaperCorpus
from autoresearch_analogy.fulltext import FullTextConfig, FullTextError
from autoresearch_analogy.observed_loop import run


MODEL = "gpt-6-loop-test"
PAPER_ID = "icml-2024/ranking"
ABSTRACT = "We improve robust ranking with group-balanced pairwise sampling."
QUOTE = "Group-balanced pairs preserve coverage of rare groups."
BODY = "Method. " + QUOTE + " We compare positive and negative examples within each group."
SOURCE = "def loss(logits, targets):\n    return binary_cross_entropy(logits, targets)\n"


def tool(name, arguments, call_id=None):
    return {"type": "function_call", "name": name, "call_id": call_id or "call_" + name,
            "arguments": json.dumps(arguments)}


def response(*calls, text=None, **overrides):
    output = list(calls)
    if text:
        output.append({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": text}]})
    value = {"id": "resp_loop", "model": MODEL, "status": "completed",
             "reasoning": {"effort": "high"}, "output": output,
             "usage": {"input_tokens": 1000, "output_tokens": 100}}
    value.update(overrides)
    return value


def report(*, mode="draft", evidence="full_text"):
    refs = [{"paper_id": PAPER_ID, "source": evidence,
             "quote": QUOTE if evidence == "full_text" else ABSTRACT}]
    if evidence == "full_text":
        refs[0]["chunk_id"] = "p1_c1"
    mechanism = {
        "bottleneck_idx": 0,
        "title": "Group-balanced pair sampling",
        "paper_ids": [PAPER_ID],
        "object_mappings": [{"source": "evaluation groups", "target": "query groups",
                             "rationale": "Both need ranking coverage despite uneven frequency."}],
        "shared_relations": "Aggregate ranking can hide rare-group mistakes.",
        "mechanism": "The source balances pair sampling across groups.",
        "intervention": "Add one group-balanced pairwise loss term.",
        "feasibility": "Training identity annotations support group membership.",
        "implementation_basis": "task" if mode == "draft" else "code",
        "code_refs": [], "runtime_evidence": [],
        "assumptions": "Group labels are observed during training.",
        "target_fit": "Training identity annotations exist; grouping coverage remains uncertain.",
        "constraints": "Preserve the fixed split and evaluator.",
        "limitations": "The paper's task differs; transfer is a hypothesis.",
        "validation_plan": "Compare validation composite AUC on the same split.",
        "rejection_criterion": "Reject if composite AUC decreases.",
        "evidence_refs": refs,
    }
    return {
        "report_schema_revision": 2,
        "bottlenecks": [{"statement": "Pointwise training may neglect group ranking.",
                         "objects": ["training loss", "group ranking metric"],
                         "relations": ["objective mismatch"],
                         "evidence": "The task evaluates subgroup AUC."}],
        "observed_facts": [{"statement": "The metric includes subgroup AUC.",
                            "source": "task", "evidence": "Task description.",
                            "code_refs": [], "runtime_evidence": []}],
        "hypotheses": ["Balanced pair sampling may improve subgroup AUC."],
        "unknowns": ["Whether the transfer improves this dataset is unknown."],
        "mechanisms": [mechanism],
    }


def abstention():
    return {"report_schema_revision": 2, "bottlenecks": [], "mechanisms": [],
            "observed_facts": [], "hypotheses": [], "unknowns": [],
            "abstention_reason": "No supported structural transfer remains."}


def document():
    return {"paper_id": PAPER_ID, "title": "Robust ranking with balanced pairs",
            "source_url": "https://example.invalid/paper.pdf",
            "pdf_sha256": "a" * 64, "text_sha256": hashlib.sha256(BODY.encode()).hexdigest(),
            "parser": {"name": "fixture"}, "page_count": 1, "warnings": [],
            "chunks": [{"chunk_id": "p1_c1", "page": 1, "section": "Method",
                        "text": BODY, "chars": len(BODY)}]}


class ScriptedEndpoint:
    """Snapshot request bodies before the loop appends more mutable messages."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def post(self, path, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["body"]))
        if not self.responses:
            raise AssertionError("Unexpected extra LLM request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    def close(self):
        self.closed = True


class AnalogyLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = PaperCorpus([
            {"id": PAPER_ID, "venue": "icml-2024", "title": "Robust ranking with balanced pairs",
             "tldr": "Balanced group pair sampling for robust ranking.", "abstract": ABSTRACT,
             "pdf_url": "https://example.invalid/paper.pdf"},
            {"id": "acl-2024/syntax", "venue": "acl-2024", "title": "Syntactic parsing",
             "tldr": "Grammar induction.", "abstract": "We study language syntax and parsing."},
            {"id": "neurips-2024/diffusion", "venue": "neurips-2024", "title": "Image diffusion",
             "tldr": "Image denoising.", "abstract": "A diffusion model synthesizes images."},
        ], {"records_sha1": "fixture"})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = FullTextConfig(enabled=True, cache_dir=str(Path(self.temp.name) / "cache"),
                                     offline=True)
        self.llm = SimpleNamespace(model=MODEL, api="responses", api_key="fixture-only",
                                   base_url="https://example.invalid/v1", reasoning_effort="high")

    def run_script(self, script, *, mode="draft", code_session=None, runtime_context=None,
                   store_error=None, max_turns=10):
        endpoint = ScriptedEndpoint(script)
        # Disable tokenizer auto-download; the loop has a conservative built-in
        # UTF-8 budget fallback. PDF downloading/parsing is tested separately.
        with patch("autoresearch_analogy.responses.make_response_client", return_value=endpoint), \
             patch("tiktoken.get_encoding", side_effect=RuntimeError("offline test")), \
             patch("autoresearch_analogy.fulltext.FullTextStore.get", side_effect=store_error,
                   return_value=(document(), False)) as get_paper:
            packet = "Task: predict toxicity; evaluate fixed subgroup AUC.\n"
            if runtime_context:
                packet += "Runtime context: " + json.dumps(runtime_context)
            result = run(packet, self.corpus, self.llm,
                         max_turns=max_turns, top_k=3, max_mechanisms=3,
                         report_char_budget=12000, mode=mode, fulltext=self.config,
                         context_options=ContextOptions(), code_session=code_session,
                         runtime_context=runtime_context, max_output_tokens=6000)
        self.assertTrue(endpoint.closed)
        return result, endpoint, get_paper

    def search(self):
        return response(tool("search_papers", {"query": "robust ranking balanced pairs"}))

    def abstract(self):
        return response(tool("read_abstract", {"ids": [PAPER_ID]}))

    def opened(self):
        return response(tool("open_paper", {"paper_id": PAPER_ID}))

    def read(self):
        return response(tool("read_paper", {"paper_id": PAPER_ID, "chunk_ids": ["p1_c1"]}))

    def test_draft_reads_full_text_and_preserves_opaque_reasoning_and_multi_tool_replay(self):
        opaque = {"type": "reasoning", "id": "rs_fixture", "encrypted_content": "opaque-fixture",
                  "summary": [], "future_field": {"keep": True}}
        initial = self.search()
        initial["output"].insert(0, opaque)
        script = [initial,
                  response(tool("read_abstract", {"ids": [PAPER_ID]}),
                           tool("open_paper", {"paper_id": PAPER_ID})),
                  self.read(), response(tool("submit_report", report()))]
        result, endpoint, get_paper = self.run_script(script)
        self.assertEqual(result.delivery_status, "accepted_complete", result.submission_attempts)
        self.assertEqual(result.paper_ids, [PAPER_ID])
        self.assertIn(QUOTE, result.report_md)
        mechanism = result.report["mechanisms"][0]
        self.assertEqual(mechanism["implementation_basis"], "task")
        self.assertEqual(mechanism["evidence_level"], "full_text")
        self.assertEqual(mechanism["evidence_refs"][0]["pdf_sha256"], "a" * 64)
        self.assertEqual(mechanism["evidence_refs"][0]["page"], 1)
        self.assertTrue(mechanism["evidence_refs"][0]["quote_verified"])
        self.assertNotIn("warning", mechanism["evidence_refs"][0])
        self.assertEqual(result.submission_attempts[0]["warnings"], [])
        self.assertEqual(result.fulltext["abstracts_read"], {PAPER_ID: ABSTRACT})
        self.assertEqual(result.fulltext["body_chars"], len(BODY))
        get_paper.assert_called_once()
        for request in endpoint.requests[1:]:
            self.assertIn(opaque, request["input"])
        second_results = [item["call_id"] for item in endpoint.requests[2]["input"]
                          if item.get("type") == "function_call_output"]
        self.assertEqual(second_results[-2:], ["call_read_abstract", "call_open_paper"])
        self.assertNotIn("opaque-fixture", json.dumps(result.model_calls))
        draft_tool_names = [tool["name"] for tool in endpoint.requests[0]["tools"]]
        self.assertNotIn("read_candidate_code", draft_tool_names)

    def test_improve_reads_frozen_source_and_cites_runtime_paths(self):
        session = CodeReadingSession([{"id": "parent1", "stage": "draft",
                                       "execution_status": "completed", "code": SOURCE}], "parent1")
        anchor = {"node_id": "parent1", "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                  "start_line": 2, "end_line": 2}
        proposal = report(mode="improve")
        proposal["mechanisms"][0].update(code_refs=[anchor], runtime_evidence=["evaluation.score"])
        proposal["observed_facts"] = [
            {"statement": "The loss uses binary cross entropy.", "source": "code",
             "evidence": "The parent source returns this objective.", "code_refs": [anchor],
             "runtime_evidence": []},
            {"statement": "The completed parent scored 0.92.", "source": "runtime",
             "evidence": "The validation summary supplies the score.", "code_refs": [],
             "runtime_evidence": ["evaluation.score"]},
        ]
        script = [response(tool("candidate_code_index", {"node_id": "parent1"}),
                           tool("read_candidate_code", {"node_id": "parent1", "start_line": 1,
                                                        "end_line": 2, "symbol": None,
                                                        "start_column": None, "max_lines": None})),
                  self.search(), self.abstract(), self.opened(), self.read(),
                  response(tool("submit_report", proposal))]
        result, _, _ = self.run_script(script, mode="improve", code_session=session,
                                       runtime_context={"evaluation": {"score": 0.92}})
        self.assertEqual(result.delivery_status, "accepted_complete", result.submission_attempts)
        mechanism = result.report["mechanisms"][0]
        self.assertEqual(mechanism["code_refs"], [anchor])
        self.assertEqual(mechanism["runtime_evidence"], ["evaluation.score"])
        self.assertTrue(any(a["start_line"] == 2 for a in result.code_reads["anchors"]))
        self.assertGreater(len(result.code_reads["ledger"]), 0)

    def test_unread_full_text_is_rejected_then_one_targeted_read_repairs_evidence(self):
        script = [self.search(), self.abstract(), self.opened(),
                  response(tool("submit_report", report(), "call_initial_submission")),
                  self.read(), response(tool("submit_report", report(), "call_corrected_submission"))]
        result, endpoint, _ = self.run_script(script)
        self.assertEqual(result.delivery_status, "accepted_complete", result.submission_attempts)
        self.assertEqual([a["status"] for a in result.submission_attempts], ["rejected", "accepted_complete"])
        first_codes = {issue["code"] for issue in result.submission_attempts[0]["issues"]}
        self.assertIn("paper_source_not_returned", first_codes)
        feedback = [json.loads(item["output"]) for item in endpoint.requests[4]["input"]
                    if item.get("call_id") == "call_initial_submission" and item.get("type") == "function_call_output"]
        self.assertEqual(feedback[0]["status"], "rejected")
        self.assertTrue(result.submission_attempts[1]["correcting"])

    def test_quote_mismatches_from_read_sources_are_delivered_with_warnings(self):
        cases = [
            ("full_text", QUOTE.replace("Group-balanced", "**Group-balanced**")),
            ("full_text", "Balanced group pairs retain representation of infrequent groups."),
            ("full_text", "This invented evidence never appeared in the paper."),
            ("abstract", "Balanced sampling improves ranking robustness across groups."),
        ]
        for source, quote in cases:
            with self.subTest(source=source, quote=quote):
                proposal = report(evidence=source)
                # Submitted verification claims must not override checks against returned text.
                proposal["mechanisms"][0]["evidence_refs"][0].update(quote=quote, quote_verified=True)
                script = [self.search(), self.abstract(), self.opened(), self.read(),
                          response(tool("submit_report", proposal))]
                result, endpoint, _ = self.run_script(script)
                self.assertEqual(result.delivery_status, "accepted_complete", result.submission_attempts)
                self.assertEqual(len(endpoint.requests), 5)
                self.assertEqual(len(result.submission_attempts), 1)
                attempt = result.submission_attempts[0]
                self.assertEqual(attempt["issues"], [])
                self.assertEqual(attempt["dropped_mechanisms"], [])
                self.assertEqual(len(attempt["warnings"]), 1)
                warning = attempt["warnings"][0]
                self.assertEqual(warning["code"], "paper_quote_not_returned")
                self.assertEqual(warning["paper_id"], PAPER_ID)
                self.assertEqual(warning["source"], source)
                self.assertIn("evidence_refs[0]", warning["location"])
                self.assertEqual(warning["chunk_id"], "p1_c1" if source == "full_text" else None)
                ref = result.report["mechanisms"][0]["evidence_refs"][0]
                self.assertEqual(ref["quote"], quote)
                self.assertFalse(ref["quote_verified"])
                self.assertEqual(ref["warning"], "paper_quote_not_returned")
                self.assertIn("source was read", result.report_md)
                self.assertIn("not verified verbatim", result.report_md)

    def test_unread_sources_still_reject_even_with_another_valid_reference(self):
        for source, read_source, extra_valid in (("full_text", False, False),
                                                ("full_text", True, False),
                                                ("full_text", True, True),
                                                ("abstract", False, False)):
            with self.subTest(source=source, read_source=read_source, extra_valid=extra_valid):
                proposal = report(evidence=source)
                script = [self.search()]
                if source == "full_text":
                    script.extend([self.abstract(), self.opened()])
                    if read_source:
                        script.append(self.read())
                        refs = proposal["mechanisms"][0]["evidence_refs"]
                        if extra_valid:
                            refs.append(copy.deepcopy(refs[0]))
                        refs[0]["chunk_id"] = "p1_unread"
                script.extend([response(tool("submit_report", proposal)),
                               response(tool("submit_report", abstention()))])
                result, _, _ = self.run_script(script)
                attempt = result.submission_attempts[0]
                self.assertEqual(attempt["status"], "rejected")
                self.assertIn("paper_source_not_returned", {i["code"] for i in attempt["issues"]})
                self.assertEqual(attempt["report"]["mechanisms"], [])

    def test_quote_warning_does_not_bypass_schema_runtime_or_code_checks(self):
        cases = ("short_quote", "long_quote", "missing_field", "runtime", "code")
        for invalid in cases:
            with self.subTest(invalid=invalid):
                mode = "improve" if invalid in {"runtime", "code"} else "draft"
                proposal = report(mode=mode)
                mechanism = proposal["mechanisms"][0]
                mechanism["evidence_refs"][0]["quote"] = "Paraphrased evidence that is absent from the returned text."
                session = None
                expected = "invalid_arguments"
                if invalid == "short_quote":
                    mechanism["evidence_refs"][0]["quote"] = "short"
                elif invalid == "long_quote":
                    mechanism["evidence_refs"][0]["quote"] = "x" * 401
                elif invalid == "missing_field":
                    del mechanism["intervention"]
                elif invalid == "runtime":
                    mechanism.update(implementation_basis="runtime", runtime_evidence=["evaluation.missing"])
                    expected = "runtime_path_unavailable"
                else:
                    session = CodeReadingSession([{"id": "parent1", "stage": "draft",
                                                   "execution_status": "completed", "code": SOURCE}], "parent1")
                    mechanism["code_refs"] = [{"node_id": "parent1",
                        "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                        "start_line": 2, "end_line": 2}]
                    expected = "code_anchor_unread"
                script = [self.search(), self.abstract(), self.opened(), self.read(),
                          response(tool("submit_report", proposal)),
                          response(tool("submit_report", abstention()))]
                result, _, _ = self.run_script(script, mode=mode, code_session=session,
                                               runtime_context={"evaluation": {"score": 0.9}})
                attempt = result.submission_attempts[0]
                self.assertEqual(attempt["status"], "rejected")
                self.assertIn(expected, {i["code"] for i in attempt["issues"]})

    def test_full_text_failure_can_fall_back_to_genuine_abstract_evidence(self):
        proposal = report(evidence="abstract")
        proposal["mechanisms"][0]["limitations"] = "The PDF is unavailable; method details remain unverified."
        script = [self.search(), self.abstract(), self.opened(), response(tool("submit_report", proposal))]
        result, _, _ = self.run_script(script, store_error=FullTextError("offline_cache_miss", "No cached PDF"))
        self.assertEqual(result.delivery_status, "accepted_complete", result.submission_attempts)
        self.assertEqual(result.report["mechanisms"][0]["evidence_level"], "abstract_only")
        self.assertEqual(result.fulltext["attempts"][PAPER_ID]["status"], "offline_cache_miss")
        self.assertEqual(result.fulltext["body_chars"], 0)

    def test_abstention_is_successful_distinct_from_missing_submission(self):
        result, _, _ = self.run_script([response(tool("submit_report", abstention()))])
        self.assertEqual(result.delivery_status, "abstained")
        self.assertFalse(result.failure_kind)
        self.assertEqual(result.report["mechanisms"], [])
        self.assertEqual(result.reason, "No supported structural transfer remains.")
        failure, _, _ = self.run_script([response(text="I have no answer."), response(text="Still no report.")])
        self.assertEqual(failure.delivery_status, "failed")
        self.assertEqual(failure.failure_kind, "missing_submission")
        self.assertIsNone(failure.report)

    def test_invalid_report_cannot_be_silently_accepted_at_turn_limit(self):
        proposal = report()
        proposal["mechanisms"][0]["evidence_refs"][0]["chunk_id"] = "p1_unread"
        script = [self.search(), response(tool("read_abstract", {"ids": [PAPER_ID]}),
                                         tool("open_paper", {"paper_id": PAPER_ID})), self.read(),
                  response(tool("submit_report", proposal)),
                  response(tool("submit_report", proposal)),
                  response(tool("submit_report", proposal))]
        result, _, _ = self.run_script(script, max_turns=6)
        self.assertEqual(result.delivery_status, "failed")
        self.assertEqual(result.failure_kind, "validation")
        self.assertFalse(result.report_md)
        self.assertEqual(len(result.submission_attempts), 3)

    def test_incomplete_response_discards_partial_tool_calls_and_reports_transport_failure(self):
        partial = response(tool("search_papers", {"query": "robust ranking"}), status="incomplete")
        result, endpoint, get_paper = self.run_script([partial])
        self.assertEqual(result.delivery_status, "failed")
        self.assertEqual(result.failure_kind, "transport")
        self.assertEqual(result.queries, [])
        self.assertEqual(len(endpoint.requests), 1)
        get_paper.assert_not_called()

    def test_chat_gateway_can_submit_without_unsupported_forced_tool_choice(self):
        client = Mock()
        message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="call_submit", function=SimpleNamespace(name="submit_report", arguments=json.dumps(abstention())))])
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100), model="claude-test")
        config = SimpleNamespace(model="claude-test", api="chat", api_key="fixture-only",
                                 base_url="https://example.invalid/v1", request_timeout=60)
        with patch("openai.OpenAI", return_value=client), \
             patch("tiktoken.get_encoding", side_effect=RuntimeError("offline test")):
            result = run("Draft task context", self.corpus, config, max_turns=1, top_k=3,
                         max_mechanisms=3, report_char_budget=12000, mode="draft",
                         fulltext=self.config, context_options=ContextOptions(), max_output_tokens=6000)
        self.assertEqual(result.delivery_status, "abstained", result.reason)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["tool_choice"], "auto")
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
