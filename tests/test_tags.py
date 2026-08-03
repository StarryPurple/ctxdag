"""Tests for directive parsing and text helpers."""

import unittest

from contextdag import (
    extract_summary,
    first_sentence,
    parse_directives,
    split_at_requires,
    strip_tags,
    summary_of,
)


class TagsTest(unittest.TestCase):
    def test_parse_refs_single_and_multi(self):
        self.assertEqual(
            parse_directives("<ref=aaaaaaaaaaaaaaaa> hello").refs,
            ("aaaaaaaaaaaaaaaa",),
        )
        self.assertEqual(
            parse_directives("<ref=aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb> hello").refs,
            ("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"),
        )

    def test_parse_requires(self):
        self.assertEqual(
            parse_directives("x<require=aaaaaaaaaaaaaaaa>y").requires,
            ("aaaaaaaaaaaaaaaa",),
        )

    def test_parse_ignores_invalid_ids(self):
        self.assertEqual(parse_directives("<ref=nothex> x").refs, ())
        self.assertEqual(parse_directives("<ref=abc> x").requires, ())

    def test_strip_tags_keeps_other_bytes(self):
        text = "a<ref=aaaaaaaaaaaaaaaa>b<require=aaaaaaaaaaaaaaaa>c"
        self.assertEqual(strip_tags(text), "abc")

    def test_split_at_requires(self):
        text = "first<require=aaaaaaaaaaaaaaaa>second<require=bbbbbbbbbbbbbbbb>third"
        segments = split_at_requires(text)
        self.assertEqual(
            segments,
            [
                ("first", "aaaaaaaaaaaaaaaa"),
                ("second", "bbbbbbbbbbbbbbbb"),
                ("third", None),
            ],
        )

    def test_split_without_requires(self):
        self.assertEqual(split_at_requires("plain"), [("plain", None)])

    def test_first_sentence(self):
        self.assertEqual(first_sentence("结论：退款。\n详细内容……"), "结论：退款。")
        self.assertEqual(first_sentence("hello world. more"), "hello world.")

    def test_summary_prefers_meta(self):
        content = "some long content that is not the summary"
        self.assertEqual(summary_of(content, {"summary": "short"}), "short")
        self.assertEqual(summary_of(content), first_sentence(content))

    def test_extract_summary_strips_first_block(self):
        body, summary = extract_summary("说明<summary>当需要换货条件时使用</summary>正文")
        self.assertEqual(body, "说明正文")
        self.assertEqual(summary, "当需要换货条件时使用")

    def test_extract_summary_no_tag(self):
        self.assertEqual(extract_summary("普通内容"), ("普通内容", None))

    def test_extract_summary_empty_ignored(self):
        text = "a<summary>  </summary>b"
        self.assertEqual(extract_summary(text), (text, None))


if __name__ == "__main__":
    unittest.main()
