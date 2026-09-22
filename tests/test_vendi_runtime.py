"""Offline runtime, evidence, embedding and numerical contracts for Vendi analysis."""
import contextlib
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autoresearch_vendi import runtime
from autoresearch_vendi.assessment import assessment_text, preserve_assessment_status
from autoresearch_vendi.changes import build_change_packet
from autoresearch_vendi.metrics import compare_samples, vendi_score


def card():
    return dict(model="linear classifier", objective="binary cross entropy", data="",
                update="gradient descent", inference="sigmoid", change="", evidence=["CHILD:1"])


def change_case():
    parent = "loss = binary_loss(logits, targets)\nloss.backward()\n"
    child = "loss = weighted_loss(logits, targets)\nloss.backward()\n"
    packet = build_change_packet(parent, child)
    result = dict(status="changed", reason="Loss computation changes before backward.", changes=[dict(
        kind="new_mechanism", description="Use weighted classification loss before gradient computation.",
        evidence=[dict(ref="PARENT:1", quote=parent.splitlines()[0]),
                  dict(ref="CHILD:1", quote=child.splitlines()[0]),
                  dict(ref="CHILD:2", quote="loss.backward()")],
        execution="connected", execution_evidence=["CHILD:2"])])
    return packet, result


def row(text="linear classifier", **values):
    return dict(text=text, extraction_status="ok", **values)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        sleep = patch.object(runtime.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_import_and_no_change_need_no_installed_model_libraries(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-B", "-S", "-c",
            "from pathlib import Path; import sys; "
            "from autoresearch_vendi.runtime import Summarizer; "
            "from autoresearch_vendi.changes import build_change_packet; "
            "s=Summarizer(Path('.')); "
            "assert s.assess_change(build_change_packet('x=1', 'x=1'))['status']=='no_change'; "
            "assert not ({'numpy','openai','torch','sentence_transformers'} & sys.modules.keys())"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_three_state_assessment_and_fast_paths_are_lazy(self):
        ask = Mock()
        summarizer = runtime.Summarizer(self.cache, ask=ask)
        unchanged = build_change_packet("x = 1\n", "# comment\nx = 1\n")
        self.assertEqual(summarizer.assess_change(unchanged)["status"], "no_change")
        unavailable = build_change_packet("x = 1\n", "x = 2\n", max_chars=1)
        self.assertEqual(summarizer.assess_change(unavailable)["status"], "insufficient_evidence")
        ask.assert_not_called()
        self.assertEqual(summarizer.calls, 0)
        self.assertIsNone(summarizer._client)
        packet, result = change_case()
        ask.return_value = json.dumps(result)
        self.assertEqual(summarizer.assess_change(packet), result)
        self.assertIn("weighted classification", assessment_text(result))

    def test_summary_cache_is_validated_and_requires_no_key(self):
        ask = Mock(return_value=json.dumps(card()))
        first = runtime.Summarizer(self.cache, ask=ask)
        self.assertEqual(first.summarize("CHILD:1: train()"), (card(), 1))
        with patch.dict(sys.modules, {"openai": None}):
            second = runtime.Summarizer(self.cache)
            self.assertEqual(second.summarize("CHILD:1: train()"), (card(), 1))
        self.assertEqual(second.calls, 0)
        self.assertEqual(ask.call_count, 1)
        path = next((self.cache / "summaries").glob("*.json"))
        path.write_text('{"model":"incomplete"}')
        with self.assertRaisesRegex(ValueError, "Invalid cached"):
            first.summarize("CHILD:1: train()")
        self.assertEqual(ask.call_count, 1)

    def test_missing_key_is_a_lazy_configuration_error_without_sdk_details(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"openai": None}):
            summarizer = runtime.Summarizer(self.cache)
            self.assertIsNone(summarizer._client)
            with self.assertRaisesRegex(ValueError, "Vendi API key is missing"):
                summarizer.summarize("CHILD:1: train()")
        self.assertEqual(summarizer.calls, 1)

    def test_source_model_endpoint_api_prompt_invalidate_summary_cache(self):
        ask = Mock(return_value=json.dumps(card()))
        for kwargs, source in [({}, "one"), ({}, "two"), ({"model":"other"}, "one"),
                               ({"base_url":"https://fixture.invalid/v1"}, "one"),
                               ({"api":"chat"}, "one")]:
            runtime.Summarizer(self.cache, ask=ask, **kwargs).summarize(source)
        with patch.object(runtime, "SUMMARY_PROMPT", runtime.SUMMARY_PROMPT + " Updated."):
            runtime.Summarizer(self.cache, ask=ask).summarize("one")
        self.assertEqual(ask.call_count, 6)

    def test_change_cache_validates_evidence_and_packet_version(self):
        packet, result = change_case()
        ask = Mock(return_value=json.dumps(result))
        summarizer = runtime.Summarizer(self.cache, ask=ask)
        summarizer.assess_change(packet)
        runtime.Summarizer(self.cache).assess_change(packet)
        self.assertEqual(ask.call_count, 1)
        updated = copy.deepcopy(packet)
        updated["version"] += "-updated"
        summarizer.assess_change(updated)
        self.assertEqual(ask.call_count, 2)
        for path in (self.cache / "changes").glob("*.json"):
            bad = copy.deepcopy(result)
            bad["changes"][0]["evidence"][0]["quote"] = "fabricated"
            path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Invalid cached"):
            summarizer.assess_change(packet)
        self.assertEqual(ask.call_count, 2)

    def test_transport_and_evidence_correction_share_three_attempts(self):
        packet, result = change_case()
        bad = copy.deepcopy(result)
        bad["changes"][0]["evidence"][0]["quote"] = "fabricated"
        ask = Mock(side_effect=[RuntimeError("secret fixture credential"), json.dumps(bad), json.dumps(result)])
        summarizer = runtime.Summarizer(self.cache, ask=ask)
        self.assertEqual(summarizer.assess_change(packet), result)
        self.assertEqual(summarizer.calls, 3)
        self.assertNotIn("secret fixture credential", ask.call_args_list[1].args[0])
        self.assertIn("does not match source", ask.call_args_list[2].args[0])
        self.assertEqual(len(list((self.cache / "changes").glob("*.json"))), 1)

    def test_final_errors_do_not_expose_provider_body_or_cache_failure(self):
        for side_effect, error_class in [(RuntimeError("secret endpoint credential"), RuntimeError),
                                         ('{"bad":"secret model output"}', ValueError)]:
            ask = Mock(side_effect=side_effect) if isinstance(side_effect, Exception) else Mock(return_value=side_effect)
            summarizer = runtime.Summarizer(self.cache, ask=ask)
            with self.assertRaises(error_class) as error:
                summarizer.summarize("CHILD:1: train()")
            self.assertNotIn("secret", str(error.exception))
            self.assertEqual(ask.call_count, 3)
            self.assertEqual(list(self.cache.rglob("*.json")), [])

    def test_nontrivial_no_change_and_unconnected_mechanisms_are_not_scored(self):
        packet, result = change_case()
        for answer in [dict(status="no_change", reason="No substantive change.", changes=[]), result]:
            if answer.get("changes"):
                answer["changes"][0].update(execution="definition_only", execution_evidence=[])
            with tempfile.TemporaryDirectory() as directory:
                summarizer = runtime.Summarizer(Path(directory), ask=lambda prompt: json.dumps(answer))
                checked = summarizer.assess_change(packet)
            self.assertEqual(checked["status"], "insufficient_evidence")
            sample = dict(text="stale", embedding=[1, 0], embedding_model="old", mechanism_card=checked)
            self.assertTrue(preserve_assessment_status(sample))
            self.assertEqual(sample["text"], "")
            self.assertNotIn("embedding", sample)

    def test_source_chunks_cover_all_characters_and_merge_one_candidate(self):
        source = "CHILD:1: " + "x" * 45 + "\nCHILD:2: " + "y" * 45
        self.assertEqual("".join(runtime.split_source(source, 30)), source)
        ask = Mock(return_value=json.dumps(card()))
        summarizer = runtime.Summarizer(self.cache, ask=ask)
        summarizer.chunk_chars = 30
        result, count = summarizer.summarize(source)
        self.assertEqual(result, card())
        self.assertGreater(count, 1)
        self.assertIn("Merge evidence for ONE candidate", ask.call_args.args[0])

    def test_modern_api_selection_and_client_settings(self):
        for api in ("responses", "chat"):
            with self.subTest(api=api):
                responses = Mock(return_value=types.SimpleNamespace(status="completed", output_text=json.dumps(card())))
                chat = Mock(return_value=types.SimpleNamespace(choices=[types.SimpleNamespace(
                    finish_reason="stop", message=types.SimpleNamespace(content=json.dumps(card())))]))
                client = types.SimpleNamespace(responses=types.SimpleNamespace(create=responses),
                    chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=chat)))
                module = types.ModuleType("openai")
                module.OpenAI = Mock(return_value=client)
                with patch.dict(sys.modules, {"openai": module}):
                    summarizer = runtime.Summarizer(self.cache / api, api=api, base_url="https://fixture.invalid/v1/", api_key="fixture")
                    module.OpenAI.assert_not_called()
                    self.assertEqual(summarizer.summarize("CHILD:1: train()")[0], card())
                module.OpenAI.assert_called_once_with(api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0, timeout=120)
                if api == "responses":
                    self.assertEqual(responses.call_args.kwargs["model"], "gpt-5.6-terra")
                    self.assertEqual(responses.call_args.kwargs["instructions"], runtime.SUMMARY_PROMPT)
                    self.assertFalse(responses.call_args.kwargs["store"])
                    chat.assert_not_called()
                else:
                    self.assertEqual(chat.call_args.kwargs["max_completion_tokens"], 4096)
                    responses.assert_not_called()


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)

    @contextlib.contextmanager
    def encoder(self, model_type="bert", hard_limit=512):
        calls = dict(created=[], encoded=[], tokenized=[])

        class Encoder:
            max_seq_length = 256

            def __init__(self, model_name, revision=None, device=None):
                calls["created"].append((model_name, revision, device))
                self.config = types.SimpleNamespace(_commit_hash=revision, model_type=model_type,
                                                    max_position_embeddings=hard_limit)

            def __getitem__(self, index):
                return types.SimpleNamespace(auto_model=types.SimpleNamespace(config=self.config))

            def tokenizer(self, text, truncation, verbose):
                calls["tokenized"].append((text, truncation, verbose))
                return {"input_ids": list(range(len(text.split())))}

            def encode(self, texts, normalize_embeddings):
                calls["encoded"].append((texts, self.max_seq_length, normalize_embeddings))
                return np.asarray([[1.0, 0.0]])

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = Encoder
        with patch.dict(sys.modules, {"sentence_transformers": module}), \
                patch("importlib.metadata.version", return_value="fixture-version"):
            yield calls

    def embed(self, samples, **kwargs):
        return runtime.embed_samples(samples, self.cache, "fixture/model", **kwargs)

    def test_cpu_revision_window_cache_and_explicit_overflow(self):
        text = " ".join("word" for _ in range(294))
        with self.encoder() as calls:
            small = row(text)
            self.embed([small], revision="one")
            self.assertEqual(small["error"], "embedding_token_limit")
            self.assertNotIn("embedding", small)
            first = row(text)
            identity = self.embed([first], revision="one", max_length=512)
            cached = row(text)
            self.assertEqual(self.embed([cached], revision="one", max_length=512), identity)
            different = row(text)
            self.assertNotEqual(self.embed([different], revision="two", max_length=512), identity)
        self.assertEqual(len(calls["encoded"]), 2)
        self.assertTrue(all(device == "cpu" for _, _, device in calls["created"]))
        self.assertTrue(all(not truncation for _, truncation, _ in calls["tokenized"]))
        self.assertEqual(identity["max_seq_length"], 512)
        self.assertEqual(first["embedding_tokens"], 294)
        self.assertEqual(first["embedding"], cached["embedding"])
        self.assertNotEqual(first["embedding_model"], different["embedding_model"])

    def test_invalid_cache_vector_and_token_window_are_rejected(self):
        with self.encoder():
            self.embed([row()], max_length=512)
            path = next((self.cache / "embeddings").glob("*.json"))
            for content in ('[0, 0]', '[NaN, 0]', 'not json'):
                path.write_text(content)
                sample = row()
                self.embed([sample], max_length=512)
                self.assertEqual(sample["extraction_status"], "error")
                self.assertNotIn("embedding", sample)
            with self.assertRaisesRegex(ValueError, "not verified"):
                self.embed([row()], max_length=513)
        with self.encoder(model_type="roberta"):
            with self.assertRaisesRegex(ValueError, "not verified"):
                self.embed([row()], max_length=512)

    def test_precomputed_vectors_are_offline_and_cannot_mix_models(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            self.assertEqual(self.embed([row(embedding=[1, 0], embedding_model="one")]),
                             dict(model="one", backend="precomputed"))
            for samples in [[row(embedding=[1, 0], embedding_model="one"), row()],
                            [row(embedding=[1, 0], embedding_model="one"), row(embedding=[1, 0], embedding_model="two")]]:
                with self.assertRaisesRegex(ValueError, "one model"):
                    self.embed(samples)
            with self.assertRaisesRegex(ValueError, "use --reembed"):
                self.embed([row(embedding=[1, 0], embedding_model="one")], max_length=512)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                self.embed([row(embedding=[1, 0], embedding_model="one"), row(embedding=[1, 0, 0], embedding_model="one")])

    def test_reembed_preserves_coverage_and_non_scoring_but_replaces_all_vectors(self):
        samples = [row(embedding=[0, 1], embedding_model="old", source_hash="unchanged"),
                   dict(text="", extraction_status="missing_source", error="missing_source"),
                   dict(text="", extraction_status="error", error="model_failed"),
                   dict(text="stale", extraction_status="ok", assessment_status="no_change", embedding=[1, 0])]
        runtime.prepare_reembedding(samples)
        self.assertNotIn("embedding", samples[0])
        self.assertEqual(samples[0]["extraction_status"], "ok")
        self.assertEqual(samples[0]["source_hash"], "unchanged")
        self.assertEqual(samples[1]["extraction_status"], "missing_source")
        self.assertEqual(samples[2]["error"], "model_failed")
        self.assertEqual(samples[3]["extraction_status"], "no_change")
        self.assertNotIn("embedding", samples[3])
        with self.encoder() as calls:
            self.embed(samples)
        self.assertEqual(samples[0]["embedding"], [1.0, 0.0])
        self.assertNotEqual(samples[0]["embedding_model"], "old")
        self.assertEqual(len(calls["encoded"]), 1)
        invalid = [row(embedding=[1, 0], embedding_model="old"), row("", embedding=[1, 0], embedding_model="old")]
        before = copy.deepcopy(invalid)
        with self.assertRaisesRegex(ValueError, "requires nonempty text"):
            runtime.prepare_reembedding(invalid)
        self.assertEqual(invalid, before)

    def test_reembed_and_embedding_never_revive_excluded_samples(self):
        excluded = [dict(text="linear classifier", extraction_status="excluded", embedding=[0, 1],
                         embedding_model="old", error="filtered"),
                    dict(text="", extraction_status="excluded", embedding=[1, 0], embedding_model="old"),
                    dict(text="stale", extraction_status="excluded", assessment_status="no_change")]
        before = copy.deepcopy(excluded)
        runtime.prepare_reembedding(excluded)
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            self.assertEqual(self.embed(excluded), {"backend":"none"})
        self.assertEqual(excluded, before)


class SolutionSummaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        sleep = patch.object(runtime.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def card(self, evidence=None):
        return {"title": "Weighted linear classifier", "summary": "Train a linear classifier with a weighted binary loss.",
                "evidence": evidence or ["SOURCE:1-2"]}

    def test_solution_cache_is_distinct_and_requires_no_key_on_hit(self):
        source = "SOURCE:1: model = Linear()\nSOURCE:2: train(model)\n"
        ask = Mock(return_value=json.dumps(self.card()))
        summarizer = runtime.Summarizer(self.cache, ask=ask)
        self.assertEqual(summarizer.summarize_solution(source), (self.card(), 1))
        self.assertEqual(len(list((self.cache / "solution_summaries").glob("*.json"))), 1)
        self.assertFalse((self.cache / "summaries").exists())
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"openai": None}):
            cached = runtime.Summarizer(self.cache)
            self.assertEqual(cached.summarize_solution(source), (self.card(), 1))
            self.assertEqual(cached.calls, 0)
        self.assertEqual(ask.call_count, 1)

    def test_complete_source_hash_invalidates_even_an_unchanged_first_fragment(self):
        prompts = []

        def ask(prompt):
            prompts.append(prompt)
            if prompt.startswith("Merge"):
                cards = json.loads(prompt.split("\n", 1)[1])
                evidence = list(dict.fromkeys(ref for card in cards for ref in card["evidence"]))
            else:
                reference = re.search(r"Available source lines: (SOURCE:\d+-\d+)", prompt).group(1)
                evidence = [reference]
            return json.dumps(self.card(evidence))

        summarizer = runtime.Summarizer(self.cache, ask=ask)
        summarizer.chunk_chars = 45
        first = "SOURCE:1: model = Linear()\nSOURCE:2: train(model)\n"
        second = first.replace("train(model)", "train(other)")
        _, chunks = summarizer.summarize_solution(first)
        first_count = len(prompts)
        summarizer.summarize_solution(second)
        self.assertGreater(chunks, 1)
        self.assertEqual(len(prompts), first_count * 2)
        self.assertEqual(prompts[0], prompts[first_count])

    def test_long_lines_are_fully_covered_and_no_diff_prompt_is_used(self):
        source = "SOURCE:1: model = " + "x" * 130 + "\nSOURCE:2: train(model)\n"
        prompts, systems = [], []
        summarizer = runtime.Summarizer(self.cache)
        summarizer.chunk_chars = 40

        def request(prompt, system_prompt):
            prompts.append(prompt)
            systems.append(system_prompt)
            if prompt.startswith("Merge"):
                cards = json.loads(prompt.split("\n", 1)[1])
                evidence = list(dict.fromkeys(ref for card in cards for ref in card["evidence"]))
            else:
                evidence = [re.search(r"Available source lines: (SOURCE:\d+-\d+)", prompt).group(1)]
            return json.dumps(self.card(evidence))

        with patch.object(summarizer, "request", side_effect=request):
            result, count = summarizer.summarize_solution(source)
        fragments = [prompt.split("\n", 2)[2] for prompt in prompts if prompt.startswith("Solution source fragment")]
        self.assertEqual("".join(fragments), source)
        self.assertEqual(count, len(fragments))
        self.assertTrue(all(system == runtime.SOLUTION_PROMPT for system in systems))
        self.assertNotIn("PARENT", "\n".join(prompts + systems))
        self.assertNotIn(runtime.CHANGE_PROMPT, systems)
        self.assertNotIn(runtime.SUMMARY_PROMPT, systems)
        self.assertEqual(result["summary"], self.card()["summary"])

    def test_bad_schema_lengths_and_references_are_rejected(self):
        invalid = [dict(self.card(), title=""), dict(self.card(), summary=" "),
                   dict(self.card(), summary="word " * 161), dict(self.card(), title="word " * 13),
                   dict(self.card(), extra="unrequested"), dict(self.card(), evidence=[])]
        invalid += [dict(self.card(), evidence=[ref]) for ref in
                    ("PARENT:1", "SOURCE:0", "SOURCE:3", "SOURCE:2-1", "SOURCE:1-999999999", "SOURCE:1-2x", 1)]
        for card in invalid:
            with self.subTest(card=card), self.assertRaises(ValueError):
                runtime.validate_solution_card(json.dumps(card), {1, 2})
        with self.assertRaises(ValueError):
            runtime.validate_solution_card(json.dumps(self.card(["SOURCE:1-3"])), {1, 3})

    def test_evidence_correction_and_cache_read_validation(self):
        source = "SOURCE:1: model = Linear()\nSOURCE:2: train(model)\n"
        ask = Mock(side_effect=[json.dumps(self.card(["SOURCE:99"])), json.dumps(self.card())])
        summarizer = runtime.Summarizer(self.cache, ask=ask)
        self.assertEqual(summarizer.summarize_solution(source)[0], self.card())
        self.assertEqual(summarizer.calls, 2)
        self.assertIn("unavailable source lines", ask.call_args.args[0])
        path = next((self.cache / "solution_summaries").glob("*.json"))
        path.write_text(json.dumps(self.card(["SOURCE:99"])))
        with self.assertRaisesRegex(ValueError, "Invalid cached"):
            summarizer.summarize_solution(source)
        self.assertEqual(ask.call_count, 2)

    def test_fragment_cannot_cite_a_line_from_another_fragment(self):
        source = "SOURCE:1: model = Linear()\nSOURCE:2: train(model)\n"
        summarizer = runtime.Summarizer(self.cache, ask=Mock(return_value=json.dumps(self.card(["SOURCE:2"]))))
        summarizer.chunk_chars = len(source.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "validation failed after 3"):
            summarizer.summarize_solution(source)
        self.assertEqual(summarizer.calls, 3)


class SolutionEmbeddingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        sleep = patch.object(runtime.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def reply(self, vector=(1.0, 0.0), model="text-embedding-3-small"):
        return types.SimpleNamespace(model=model, data=[types.SimpleNamespace(index=0, embedding=list(vector))])

    @contextlib.contextmanager
    def client(self, side_effect=None):
        create = Mock(side_effect=side_effect) if side_effect is not None else Mock(return_value=self.reply())
        client = types.SimpleNamespace(embeddings=types.SimpleNamespace(create=create), close=Mock())
        module = types.ModuleType("openai")
        module.OpenAI = Mock(return_value=client)
        with patch.dict(sys.modules, {"openai": module}):
            yield module.OpenAI, create, client

    def embed(self, samples, **kwargs):
        return runtime.embed_solution_samples(samples, self.cache, **kwargs)

    def test_duplicate_candidates_keep_frequency_and_cache_needs_no_key(self):
        samples = [row("weighted classifier", candidate_id="one"), row("weighted classifier", candidate_id="two")]
        untouched = [dict(text="excluded", extraction_status="excluded", embedding=[0, 1]),
                     dict(text="failed", extraction_status="error", error="source_failed")]
        before = copy.deepcopy(untouched)
        with self.client() as (factory, create, client):
            identity = self.embed(samples + untouched, api_key="fixture", base_url="https://fixture.invalid/v1/")
            self.assertEqual(create.call_count, 1)
            create.assert_called_once_with(model="text-embedding-3-small", input=["weighted classifier"], encoding_format="float")
            factory.assert_called_once_with(api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0, timeout=120)
            client.close.assert_called_once()
        self.assertEqual(len(samples), 2)
        self.assertEqual([sample["candidate_id"] for sample in samples], ["one", "two"])
        self.assertEqual(samples[0]["embedding"], samples[1]["embedding"])
        self.assertEqual(untouched, before)
        self.assertEqual(identity["dimensions"], 2)
        self.assertNotIn("fixture.invalid", json.dumps(identity))
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"openai": None}):
            repeated = [row("weighted classifier")]
            self.assertEqual(self.embed(repeated, base_url="https://fixture.invalid/v1"), identity)
        self.assertEqual(repeated[0]["embedding_model"], samples[0]["embedding_model"])

    def test_endpoint_model_and_text_changes_invalidate_embedding_cache(self):
        with self.client(side_effect=[self.reply(), self.reply(), self.reply(model="other-model"), self.reply()]) as (_, create, _):
            initial = self.embed([row("first")], api_key="fixture")
            endpoint = self.embed([row("first")], base_url="https://fixture.invalid/v1", api_key="fixture")
            model = self.embed([row("first")], model_name="other-model", api_key="fixture")
            self.embed([row("second")], api_key="fixture")
        self.assertEqual(create.call_count, 4)
        self.assertNotEqual(initial, endpoint)
        self.assertNotEqual(initial, model)

    def test_precomputed_is_offline_rejects_mixed_models_and_dimension_changes(self):
        with patch.dict(sys.modules, {"openai": None}):
            self.assertEqual(self.embed([row(embedding=[1, 0], embedding_model="one")]),
                             {"backend":"precomputed", "model":"one"})
            for samples in [[row(embedding=[1, 0], embedding_model="one"), row()],
                            [row(embedding=[1, 0], embedding_model="one"), row(embedding=[1, 0], embedding_model="two")]]:
                with self.assertRaisesRegex(ValueError, "one model"):
                    self.embed(samples)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                self.embed([row(embedding=[1, 0], embedding_model="one"), row(embedding=[1, 0, 0], embedding_model="one")])

    def test_cached_invalid_vectors_never_trigger_an_api_or_enter_scores(self):
        with self.client():
            self.embed([row()], api_key="fixture")
        path = next((self.cache / "solution_embeddings").glob("*.json"))
        with patch.dict(sys.modules, {"openai": None}):
            for content in ("not json", "[0, 0]", "[NaN, 1]", "[Infinity, 1]", "[[1], [0]]"):
                path.write_text(content)
                sample = row()
                self.embed([sample])
                self.assertEqual(sample["extraction_status"], "error")
                self.assertNotIn("embedding", sample)

    def test_transport_attempts_are_bounded_and_provider_errors_are_private(self):
        error = RuntimeError("secret provider body and credentials")
        with self.client(side_effect=[error, error, self.reply()]) as (_, create, _):
            self.embed([row("retry succeeds")], api_key="fixture")
            self.assertEqual(create.call_count, 3)
        with self.client(side_effect=error) as (_, create, client):
            with self.assertRaises(RuntimeError) as caught:
                self.embed([row("retry fails")], api_key="fixture")
            self.assertEqual(create.call_count, 3)
            self.assertNotIn("secret", str(caught.exception))
            client.close.assert_called_once()

    def test_response_model_dimensions_and_missing_key_are_validated(self):
        with self.client(side_effect=[self.reply(model="wrong")]):
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.embed([row()], api_key="fixture")
        with self.client(side_effect=[self.reply(), self.reply((1.0, 0.0, 0.0))]):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                self.embed([row("first"), row("second")], api_key="fixture")
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"openai": None}):
            with self.assertRaisesRegex(ValueError, "embedding API key is missing"):
                self.embed([row("not cached")])

    def test_http_404_is_reported_safely_and_is_not_retried(self):
        class NotFoundError(Exception):
            status_code = 404

        error = NotFoundError("secret provider body and credentials")
        with self.client(side_effect=error) as (_, create, client):
            with self.assertRaises(runtime.EmbeddingRequestError) as caught:
                self.embed([row("missing endpoint")], api_key="fixture")
            create.assert_called_once()
            client.close.assert_called_once()
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertIn("NotFoundError; HTTP 404; attempts=1", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_real_sdk_embedding_shape_over_mock_http(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("OpenAI SDK is not installed in this offline test environment")
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"object":"list", "model":"text-embedding-3-small",
                "data":[{"object":"embedding", "index":0, "embedding":[1.0, 0.0]}],
                "usage":{"prompt_tokens":3, "total_tokens":3}})

        client = OpenAI(api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        with patch("openai.OpenAI", return_value=client):
            samples = [row("weighted classifier")]
            result = self.embed(samples, api_key="fixture", base_url="https://fixture.invalid/v1")
        self.assertEqual(result["backend"], "openai")
        self.assertEqual(samples[0]["embedding"], [1.0, 0.0])
        self.assertEqual(requests, [{"input":["weighted classifier"], "model":"text-embedding-3-small", "encoding_format":"float"}])


class NumericalTests(unittest.TestCase):
    def test_identical_orthogonal_frequency_and_invalid_vectors(self):
        self.assertAlmostEqual(vendi_score([[1, 2], [2, 4], [3, 6]]), 1)
        self.assertAlmostEqual(vendi_score(np.eye(5)), 5)
        expected = math.exp(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
        self.assertAlmostEqual(vendi_score([[1, 0]] * 3 + [[0, 1]]), expected)
        self.assertAlmostEqual(vendi_score([[1e300, 0], [0, 1e-300]]), 2)
        for vectors in ([], [[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1], [1, 2]]):
            with self.assertRaises(ValueError):
                vendi_score(vectors)

    def test_equal_size_paired_comparison_and_missing_data(self):
        def samples(run_id, arm, vectors, pair_id="pair"):
            return [dict(task="task", run_id=run_id, arm=arm, pair_id=pair_id,
                         stage="improve", view="implementation", embedding=vector) for vector in vectors]
        values = (samples("b", "baseline", [[1, 0]] * 4)
                  + samples("a", "analogy", [[1, 0], [0, 1]]))
        scores, comparisons = compare_samples(values, baseline="baseline")
        self.assertEqual({sample["m"] for sample in scores}, {2})
        self.assertTrue(all(abs(sample["delta"] - 1) < 1e-12 for sample in comparisons))
        self.assertEqual((scores, comparisons), compare_samples(list(reversed(values)), baseline="baseline"))
        _, missing = compare_samples(samples("a", "analogy", [[1, 0], [0, 1]]), baseline="baseline")
        self.assertTrue(all(sample["delta"] is None for sample in missing))


if __name__ == "__main__":
    unittest.main()
