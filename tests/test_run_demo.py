"""Tests for the demo's chat-response parsing (real errors, not KeyError)."""

import importlib.util
import sys
import unittest

import requests


def load_run_demo():
    spec = importlib.util.spec_from_file_location(
        "run_demo", "demo/run_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, body, status=200, text=None, non_json=False):
        self._body = body
        self.status_code = status
        self.text = text if text is not None else repr(body)
        self.non_json = non_json

    def json(self):
        if self.non_json:
            raise requests.exceptions.JSONDecodeError("no json", "doc", 0)
        return self._body


class ParseChatResponseTest(unittest.TestCase):
    def setUp(self):
        self.parse = load_run_demo().parse_chat_response

    def test_valid_choices(self):
        resp = _FakeResponse(
            {"choices": [{"message": {"content": "  回复内容  "}}]}
        )
        self.assertEqual(self.parse(resp, "test"), "回复内容")

    def test_non_json_raises_with_status(self):
        resp = _FakeResponse(None, status=502, text="<html>bad gateway</html>", non_json=True)
        with self.assertRaises(RuntimeError) as cm:
            self.parse(resp, "test")
        self.assertIn("502", str(cm.exception))

    def test_error_key_surfaced(self):
        resp = _FakeResponse({"error": {"message": "boom"}})
        with self.assertRaises(RuntimeError) as cm:
            self.parse(resp, "test")
        self.assertIn("boom", str(cm.exception))

    def test_fastapi_detail_surfaced(self):
        resp = _FakeResponse({"detail": "Internal Server Error"}, status=500)
        with self.assertRaises(RuntimeError) as cm:
            self.parse(resp, "test")
        self.assertIn("Internal Server Error", str(cm.exception))
        self.assertIn("500", str(cm.exception))

    def test_missing_choices_reports_body_keys(self):
        resp = _FakeResponse({}, status=500)
        with self.assertRaises(RuntimeError) as cm:
            self.parse(resp, "test")
        self.assertIn("no 'choices'", str(cm.exception))

    def test_non_dict_body(self):
        resp = _FakeResponse(["not", "a", "dict"])
        with self.assertRaises(RuntimeError):
            self.parse(resp, "test")


if __name__ == "__main__":
    unittest.main()
