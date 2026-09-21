"""Offline transport contract tests; never contact an LLM endpoint."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import openai

from autoresearch_analogy import model_profiles
from autoresearch_analogy.responses import (
    ResponsesError,
    make_response_client,
    request_response,
    response_function_calls,
    response_info,
    response_text,
    should_retry_outer,
    uses_responses,
)


MODEL = "gpt-6-test"


def completed(**overrides):
    response = {
        "id": "resp_test",
        "status": "completed",
        "model": MODEL,
        "reasoning": {"effort": "high"},
        "output": [],
        "usage": {"input_tokens": 17, "output_tokens": 23, "total_tokens": 40},
    }
    response.update(overrides)
    return response


def api_error(status, code="server_error", secret="do-not-log-this-key"):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid/responses"))
    return openai.APIStatusError(secret, response=response,
                                 body={"error": {"code": code, "message": secret}})


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class ResponsesTransportTests(unittest.TestCase):
    def request(self, client, **kwargs):
        return request_response(client, model=MODEL, input_items=[], retry_delay=0, **kwargs)

    def test_explicit_api_overrides_model_routing(self):
        self.assertTrue(uses_responses("provider/gpt-6"))
        self.assertTrue(uses_responses("gpt-5.6-sol-2026-09-01"))
        self.assertFalse(uses_responses("gpt-60"))
        self.assertFalse(uses_responses("gpt-6", "chat"))
        self.assertTrue(uses_responses("gateway/model-alias", "responses"))
        with self.assertRaises(ResponsesError):
            uses_responses(MODEL, "unknown")

    def test_client_owns_retry_and_reads_configured_transport_timeout(self):
        config = SimpleNamespace(api_key="secret", base_url="https://example.invalid/v1",
                                 request_timeout=42)
        with patch("autoresearch_analogy.responses.OpenAI") as factory:
            make_response_client(config)
        factory.assert_called_once_with(api_key="secret", base_url=config.base_url,
                                        timeout=42, max_retries=0)

    def test_real_sdk_preserves_raw_json_and_sse_responses(self):
        # Exercise the locked SDK parser; mocking client.post hides cast_to bugs.
        response = completed(
            output=[
                {"type": "reasoning", "encrypted_content": "opaque-state",
                 "unknown_future_field": {"values": [None, True, 1, "1"]}},
                {"type": "function_call", "call_id": "call_search",
                 "name": "search_papers", "arguments": '{"query":"ranking"}'},
            ],
            future_metadata={"preserve": ["unknown", {"nested": True}]},
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                requests = []
                responses = []

                def handler(request):
                    requests.append(request)
                    if stream:
                        event = {"type": "response.completed", "response": response}
                        reply = httpx.Response(200, headers={"content-type": "text/event-stream"},
                                               content="event: response.completed\ndata: "
                                               + json.dumps(event) + "\n\ndata: [DONE]\n\n")
                    else:
                        reply = httpx.Response(200, json=response)
                    responses.append(reply)
                    return reply

                with openai.OpenAI(api_key="offline-test-key", base_url="https://example.invalid/v1",
                                   max_retries=0, http_client=httpx.Client(
                                       transport=httpx.MockTransport(handler))) as client:
                    result = self.request(client, stream=stream)
                self.assertEqual(result, response)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].url.path, "/v1/responses")
                self.assertEqual(json.loads(requests[0].content).get("stream", False), stream)
                self.assertTrue(responses[0].is_closed)

    def test_real_sdk_raw_json_still_requires_an_object(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=["not an object"]))
        with openai.OpenAI(api_key="offline-test-key", base_url="https://example.invalid/v1",
                           max_retries=0, http_client=httpx.Client(transport=transport)) as client:
            with self.assertRaises(ResponsesError) as caught:
                self.request(client)
        self.assertEqual(caught.exception.category, "protocol")
        self.assertEqual(caught.exception.attempts, 1)

    def test_complete_replay_preserves_opaque_reasoning_and_multiple_tools(self):
        output = [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque-state",
             "summary": [], "unknown_future_field": {"preserve": True}},
            {"type": "function_call", "id": "fc_1", "call_id": "call_search",
             "name": "search_papers", "arguments": '{"query":"robust ranking"}'},
            {"type": "function_call", "id": "fc_2", "call_id": "call_read",
             "name": "read_abstract", "arguments": '{"paper_ids":["paper/1"]}'},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Read evidence."}]},
        ]
        response = completed(output=output)
        client = Mock()
        client.post.side_effect = [response, completed()]
        first = self.request(client)
        self.assertIs(first, response)
        self.assertEqual(response_function_calls(first), output[1:3])
        self.assertEqual(response_text(first), "Read evidence.")
        # This is the caller contract used by the observed tool loop. Replaying
        # only the visible assistant text would lose both tool IDs and reasoning.
        replay = [{"role": "user", "content": "diagnose"}] + first["output"] + [
            {"type": "function_call_output", "call_id": "call_search", "output": "[]"},
            {"type": "function_call_output", "call_id": "call_read", "output": "{}"},
        ]
        before = copy.deepcopy(replay)
        request_response(client, model=MODEL, input_items=replay)
        body = client.post.call_args.kwargs["body"]
        self.assertEqual(body["input"], before)
        self.assertFalse(body["store"])
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(replay, before)

    def test_explicit_responses_model_alias_is_bound_to_returned_model(self):
        client = Mock()
        client.post.return_value = completed(model="custom-reasoner")
        result = request_response(client, model="custom-reasoner", input_items=[])
        self.assertEqual(result["model"], "custom-reasoner")

    def test_tool_schema_is_converted_without_mutating_input(self):
        tool = {"type": "function", "function": {
            "name": "read_paper", "description": "Read a paper",
            "parameters": {"type": "object", "properties": {"paper_id": {"type": "string"}}},
        }}
        before = copy.deepcopy(tool)
        client = Mock()
        client.post.return_value = completed()
        self.request(client, tools=[tool], tool_choice={"type": "function", "function": {"name": "read_paper"}})
        body = client.post.call_args.kwargs["body"]
        self.assertEqual(body["tools"][0]["name"], "read_paper")
        self.assertFalse(body["tools"][0]["strict"])
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["tool_choice"], {"type": "function", "name": "read_paper"})
        self.assertEqual(tool, before)

    def test_partial_response_model_effort_and_protocol_errors_are_terminal(self):
        cases = [
            (completed(status="incomplete", output=[{"type": "function_call"}]), "incomplete"),
            (completed(model="a-different-model"), "model_mismatch"),
            (completed(reasoning={"effort": "low"}), "reasoning_mismatch"),
            (completed(output=[{"type": "function_call", "name": "read_paper", "arguments": "{}"}]), "protocol"),
            (completed(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]), "refusal"),
            (completed(output=[{"type": "message", "content": "malformed"}]), "protocol"),
            (completed(reasoning="malformed"), "protocol"),
        ]
        for response, category in cases:
            with self.subTest(category=category, response=response):
                client = Mock()
                client.post.return_value = response
                with self.assertRaises(ResponsesError) as caught:
                    self.request(client)
                self.assertEqual(caught.exception.category, category)
                self.assertEqual(client.post.call_count, 1)
                self.assertFalse(should_retry_outer(caught.exception))

    def test_model_snapshot_suffix_is_accepted(self):
        client = Mock()
        client.post.return_value = completed(model=MODEL + "-2026-09-01")
        self.assertEqual(self.request(client)["status"], "completed")

    def test_stream_uses_terminal_object_and_closes(self):
        final = completed(output=[{"type": "message", "content": [{"type": "output_text", "text": "complete"}]}])
        stream = FakeStream([
            {"type": "response.output_text.delta", "delta": "partial"},
            {"event": "response.completed", "data": {"response": final}},
        ])
        client = Mock()
        client.post.return_value = stream
        self.assertIs(self.request(client, stream=True), final)
        self.assertTrue(stream.closed)

    def test_stream_without_terminal_event_retries_but_discards_partial_tools(self):
        first = FakeStream([{"type": "response.function_call_arguments.delta", "delta": "partial"}])
        second = FakeStream([{"type": "response.completed", "response": completed()}])
        client = Mock()
        client.post.side_effect = [first, second]
        self.assertEqual(self.request(client, stream=True)["output"], [])
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(client.post.call_count, 2)

    def test_stream_error_event_does_not_leak_provider_messages(self):
        stream = FakeStream([{"type": "error", "error": {"code": "do-not-log-this-key", "message": "do-not-log-this-key"}}])
        client = Mock()
        client.post.return_value = stream
        with self.assertRaises(ResponsesError) as caught:
            self.request(client, stream=True)
        self.assertNotIn("do-not-log-this-key", str(caught.exception))
        self.assertTrue(stream.closed)
        self.assertEqual(client.post.call_count, 1)

    def test_transient_retry_is_bounded_even_when_caller_requests_more(self):
        client = Mock()
        client.post.side_effect = api_error(503)
        with patch("autoresearch_analogy.responses.time.sleep") as sleep:
            with self.assertRaises(ResponsesError) as caught:
                self.request(client, max_attempts=20)
        self.assertEqual(client.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(caught.exception.category, "transient_exhausted")
        self.assertEqual(caught.exception.attempts, 3)
        self.assertNotIn("do-not-log-this-key", str(caught.exception))

    def test_quota_or_auth_failure_is_not_retried(self):
        for status, code in [(429, "insufficient_quota"), (401, "invalid_api_key")]:
            with self.subTest(status=status, code=code):
                client = Mock()
                client.post.side_effect = api_error(status, code)
                with self.assertRaises(ResponsesError) as caught:
                    self.request(client)
                self.assertEqual(client.post.call_count, 1)
                self.assertEqual(caught.exception.category, "request_error")
                self.assertNotIn("do-not-log-this-key", str(caught.exception))

    def test_retryable_error_in_json_body_retries(self):
        client = Mock()
        client.post.side_effect = [completed(error={"code": "overloaded"}), completed()]
        self.assertEqual(self.request(client)["status"], "completed")
        self.assertEqual(client.post.call_count, 2)

    def test_error_metadata_excludes_messages_reasoning_and_unknown_usage_fields(self):
        secret = "do-not-log-this-key"
        response = completed(
            status="incomplete",
            error={"code": secret, "message": secret},
            output=[{"type": "reasoning", "encrypted_content": secret, "summary": secret}],
            incomplete_details={"reason": "max_output_tokens", "message": secret},
            usage={"input_tokens": 2, "output_tokens": 3, "debug": secret,
                   "output_tokens_details": {"reasoning_tokens": 1, "debug": secret}},
        )
        self.assertNotIn(secret, json.dumps(response_info(response)))
        client = Mock()
        client.post.return_value = response
        with self.assertRaises(ResponsesError) as caught:
            self.request(client)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, json.dumps(caught.exception.response_info))


class ModelProfileTests(unittest.TestCase):
    def test_gateway_prefix_is_normalized_for_chat_parameters(self):
        self.assertTrue(model_profiles.is_openai_reasoning_model("provider/gpt-5-test"))
        self.assertTrue(model_profiles.uses_max_completion_tokens("provider/gpt-5-test"))
        self.assertFalse(model_profiles.supports_sampling_params("provider/gpt-5-test"))
        self.assertFalse(model_profiles.uses_max_completion_tokens("anthropic/claude-test"))


if __name__ == "__main__":
    unittest.main()
