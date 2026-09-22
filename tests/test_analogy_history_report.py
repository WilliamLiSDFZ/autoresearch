"""Offline contracts for history-aware reports; no model or training calls."""

import copy
import unittest
from types import SimpleNamespace

from jsonschema import validate as validate_schema

from autoresearch_analogy import agent, report_v2
from autoresearch_analogy.corpus import PaperCorpus


PAPER_ID = "icml-2024/ranking"


class HistoryReportTests(unittest.TestCase):
    def setUp(self):
        self.corpus = PaperCorpus([
            {"id": PAPER_ID, "venue": "icml-2024", "title": "Balanced ranking",
             "tldr": "Balance pair sampling.", "abstract": "Balanced pairs improve ranking."}
        ], {"records_sha1": "fixture"})
        self.runtime = {
            "public_validation": {"score": 0.8},
            "experiment_history": [
                {"trial_id": "trial0001", "execution_status": "completed", "validation": "verified",
                 "metrics": {"score": 0.8}},
                {"trial_id": "trial0002", "execution_status": "crash", "validation": "pending"},
                {"trial_id": "trial0003", "execution_status": "incomplete", "validation": "pending"},
            ],
        }
        self.proposal = {
            "report_schema_revision": 2,
            "bottlenecks": [{"statement": "Rare groups have weaker ranking.",
                             "evidence": "The visible validation score leaves room for improvement."}],
            "observed_facts": [], "hypotheses": [], "unknowns": [],
            "mechanisms": [{
                "bottleneck_idx": 0, "title": "Balanced pair sampling", "paper_ids": [PAPER_ID],
                "object_mappings": [], "shared_relations": "Rare groups need positive-negative pairs.",
                "mechanism": "Sample rare positive-negative pairs.",
                "intervention": "Use a smaller pairwise coefficient and a separate ranking microbatch.",
                "feasibility": "Training group labels are available.",
                "implementation_basis": "runtime", "code_refs": [],
                "runtime_evidence": ["public_validation.score"],
                "assumptions": "Training groups contain both classes.",
                "target_fit": "Ranking groups are defined by the task.",
                "constraints": "Keep the fixed evaluator and validation split.",
                "validation_plan": "Compare the same-seed composite score and runtime.",
                "rejection_criterion": "Reject if the composite score falls.",
            }],
        }

    def validate(self, proposal=None, runtime=None, mode="improve"):
        return report_v2.validate_detailed(
            self.proposal if proposal is None else proposal, {PAPER_ID}, self.corpus, 3,
            reading=None, abstracts={}, code_session=None,
            runtime_context=self.runtime if runtime is None else runtime, mode=mode)

    def comparison(self, related=None):
        return {
            "related_trial_ids": ["trial0002"] if related is None else related,
            "difference": "Keep ranking rows in a separate forward and lower its coefficient.",
            "retry_reason": "The earlier run crashed before completing evaluation; efficacy remains unmeasured.",
        }

    def test_schema_keeps_history_optional_for_legacy_calls(self):
        tools = report_v2.extend_tools(agent.TOOLS, mode="improve")
        mechanism = tools[-1]["function"]["parameters"]["properties"]["mechanisms"]["items"]
        self.assertNotIn("history_comparison", mechanism["required"])
        schema = mechanism["properties"]["history_comparison"]
        self.assertEqual(schema["required"], ["related_trial_ids", "difference", "retry_reason"])
        self.assertNotIn("maxLength", schema["properties"]["difference"])
        self.assertNotIn("history_comparison", agent.TOOLS[-1]["function"]["parameters"]["properties"]["mechanisms"]["items"]["properties"])

    def test_no_history_preserves_existing_report_contract(self):
        for history in (None, []):
            with self.subTest(history=history):
                runtime = {"public_validation": {"score": 0.8}}
                if history is not None:
                    runtime["experiment_history"] = history
                details = self.validate(runtime=runtime)
                self.assertEqual(details["issues"], [])
                self.assertEqual(len(details["report"]["mechanisms"]), 1)
                self.assertNotIn("history_comparison", details["report"]["mechanisms"][0])

    def test_visible_history_requires_comparison_without_rejecting_shared_facts(self):
        details = self.validate()
        self.assertTrue(details["shared_facts_valid"])
        self.assertEqual(details["report"]["mechanisms"], [])
        self.assertEqual(details["issues"][0]["code"], "history_comparison_required")
        self.assertEqual(details["dropped_mechanisms"][0]["original_index"], 0)

    def test_completed_crashed_and_incomplete_trials_are_all_comparable(self):
        for trial_id in ("trial0001", "trial0002", "trial0003"):
            with self.subTest(trial_id=trial_id):
                proposal = copy.deepcopy(self.proposal)
                comparison = self.comparison([trial_id])
                comparison["retry_reason"] = "A lower dose and separate forward address the earlier tradeoff."
                proposal["mechanisms"][0]["history_comparison"] = comparison
                details = self.validate(proposal)
                self.assertEqual(details["issues"], [])
                self.assertEqual(details["report"]["mechanisms"][0]["history_comparison"], comparison)

    def test_unknown_trial_rejected_without_hiding_other_valid_mechanisms(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["mechanisms"][0]["history_comparison"] = self.comparison(["trial9999"])
        second = copy.deepcopy(proposal["mechanisms"][0])
        second["title"] = "Bounded retry"
        second["history_comparison"] = self.comparison()
        proposal["mechanisms"].append(second)
        details = self.validate(proposal)
        self.assertEqual(details["issues"][0]["code"], "history_trial_unavailable")
        self.assertEqual(details["mechanism_mapping"], [{"original_index": 1, "mechanism_id": "m1"}])
        self.assertEqual(details["report"]["mechanisms"][0]["title"], "Bounded retry")

    def test_history_comparison_requires_types_and_necessary_explanations(self):
        cases = [
            (None, "history_comparison_required"),
            ({**self.comparison(), "related_trial_ids": "trial0002"}, "history_trial_ids_type"),
            ({**self.comparison(), "related_trial_ids": [2]}, "history_trial_ids_type"),
            ({**self.comparison(), "difference": " "}, "history_difference_required"),
            ({**self.comparison(), "difference": 3}, "history_difference_required"),
            ({**self.comparison(), "retry_reason": " "}, "history_retry_reason_required"),
            ({**self.comparison([]), "retry_reason": None}, "history_retry_reason_required"),
        ]
        for comparison, code in cases:
            with self.subTest(comparison=comparison):
                proposal = copy.deepcopy(self.proposal)
                proposal["mechanisms"][0]["history_comparison"] = comparison
                details = self.validate(proposal)
                self.assertIn(code, [issue["code"] for issue in details["issues"]])
                self.assertEqual(details["report"]["mechanisms"], [])

    def test_new_mechanism_can_explain_no_related_attempt(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["mechanisms"][0]["history_comparison"] = {
            "related_trial_ids": [], "difference": "No visible attempt changes this sampling component.",
            "retry_reason": "",
        }
        details = self.validate(proposal)
        self.assertEqual(details["issues"], [])
        _, text = report_v2.render(details["report"], self.corpus, 0)
        self.assertIn("related trials: none", text)
        self.assertNotIn("Retry rationale:", text)

    def test_only_visible_historical_metrics_can_support_runtime_facts(self):
        for path, valid in (("experiment_history.0.metrics.score", True),
                            ("experiment_history.1.metrics.score", False),
                            ("experiment_history.0.adoption.reason", False)):
            with self.subTest(path=path):
                proposal = copy.deepcopy(self.proposal)
                proposal["mechanisms"][0]["history_comparison"] = self.comparison()
                proposal["observed_facts"] = [{
                    "statement": "An earlier score is available.", "source": "runtime",
                    "evidence": "Verified historical evaluation.", "runtime_evidence": [path], "code_refs": [],
                }]
                details = self.validate(proposal)
                self.assertEqual(details["shared_facts_valid"], valid)
                if not valid:
                    self.assertIn("runtime_path_unavailable", [issue["code"] for issue in details["issues"]])

    def test_zero_budget_preserves_long_comparison_and_every_mechanism(self):
        proposal = copy.deepcopy(self.proposal)
        comparison = self.comparison()
        comparison["difference"] += " Observed implementation detail." * 800
        proposal["mechanisms"][0]["history_comparison"] = comparison
        second = copy.deepcopy(proposal["mechanisms"][0])
        second["title"] = "Second mechanism"
        proposal["mechanisms"].append(second)
        details = self.validate(proposal)
        self.assertEqual(details["issues"], [])
        retained, text = report_v2.render(details["report"], self.corpus, 0)
        self.assertEqual(len(retained["mechanisms"]), 2)
        self.assertEqual(text.count("**History comparison**"), 2)
        self.assertIn(comparison["difference"], text)
        self.assertIn("Retry rationale: " + comparison["retry_reason"], text)
        bounded, text = report_v2.render(details["report"], self.corpus, 100)
        self.assertEqual(bounded["mechanisms"], [])
        self.assertEqual(text, "")

    def test_zero_budget_in_legacy_renderer_preserves_multiple_long_mechanisms(self):
        clean = self.validate(runtime={"public_validation": {"score": 0.8}})["report"]
        clean["mechanisms"][0]["intervention"] = "Concrete intervention. " * 800
        second = copy.deepcopy(clean["mechanisms"][0])
        second["title"] = "Second mechanism"
        clean["mechanisms"].append(second)
        text = agent.render_report(clean, self.corpus, 0)
        self.assertIn("### Balanced pair sampling", text)
        self.assertIn("### Second mechanism", text)
        self.assertEqual(text.count("Concrete intervention."), 1600)

    def test_long_prose_passes_schemas_and_is_preserved_by_both_validators(self):
        proposal = copy.deepcopy(self.proposal)
        mechanism = proposal["mechanisms"][0]
        fields = ("assumptions", "target_fit", "limitations", "validation_plan",
                  "constraints", "rejection_criterion")
        long_text = "Concrete evidence and transfer conditions. " * 60
        self.assertGreater(len(long_text), 1500)
        for field in fields:
            mechanism[field] = long_text.strip()
        abstract = self.corpus.by_id[PAPER_ID]["abstract"]
        mechanism["evidence_refs"] = [{"paper_id": PAPER_ID, "source": "abstract", "quote": abstract}]
        mechanism["history_comparison"] = self.comparison()
        legacy_tools = agent.reading_tools()
        for tools in (legacy_tools, report_v2.extend_tools(legacy_tools, mode="improve")):
            schema = next(tool["function"]["parameters"] for tool in tools
                          if tool["function"]["name"] == "submit_report")
            validate_schema(proposal, schema)
        reading = SimpleNamespace()
        legacy, issues = agent.validate_report(proposal, {PAPER_ID}, self.corpus, 3,
                                              reading=reading, abstracts={PAPER_ID: abstract})
        self.assertEqual(issues, [])
        for field in fields[:4]:
            self.assertEqual(legacy["mechanisms"][0][field], long_text.strip())
        details = report_v2.validate_detailed(proposal, {PAPER_ID}, self.corpus, 3,
            reading=reading, abstracts={PAPER_ID: abstract}, code_session=None,
            runtime_context=self.runtime, mode="improve")
        self.assertEqual(details["issues"], [])
        for field in fields:
            self.assertEqual(details["report"]["mechanisms"][0][field], long_text.strip())


if __name__ == "__main__":
    unittest.main()
