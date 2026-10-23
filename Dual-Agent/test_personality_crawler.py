import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from personality_crawler import (ANSWER_SCORES, CrawlerError, PersonalityCrawler,
    attach_reports, build_payload, main, render_report, validate_answers)
from questionnaire_agent import HERE, QuestionnaireAgent, export_proxy, fingerprint, read_json
from test_questionnaire_agent import ScriptedTransport

BANK = read_json(HERE / "questionnaire.json")
REPORT = {
    "content": {
        "intro": {"traitsTitle": "INTJ-A", "parts": ["<p>Independent &amp; reflective.</p>"]},
        "career": {"title": "Career", "intro": "<p>Build understanding.</p>",
                   "strengths": [{"title": "Focus", "subtitle": "Sustained attention."}]},
        "personalGrowth": {"title": "Personal Growth"},
        "relationships": {"title": "Relationships", "traits": [{"title": "Direct communication"}]}
    },
    "personality": {"traits": [{"trait": "Energy", "pct": 65, "label": "Introverted",
                                "snippet": "Prefers quiet settings."}]}
}


class Response:
    def __init__(self, value=None, status=200, headers=None):
        self.value, self.status_code, self.headers = value, status, headers or {}

    def json(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers, self.cookies, self.calls = {}, {}, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class CrawlerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("ODREC_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)

    def crawler(self, responses=None):
        session = Session(responses or [Response(), Response({}), Response(REPORT)])
        return PersonalityCrawler(session, sleep=lambda seconds: None), session

    def agent_output(self):
        return {"status": "complete", "subject_id": "A", "kernels": {
            role: {"answer_string": str(value) * 60, "item_responses": [value] * 60,
                   "questionnaire_sha256": fingerprint(BANK)}
            for role, value in [("self", 2), ("partner", 5)]},
            "semantic_texts": {"self_kernel": "self items", "partner_kernel": "partner items"}}

    def test_agent_exports_exactly_sixty_digits(self):
        replies = []
        expected = []
        for question in BANK["questions"]:
            value = (question["id"] - 1) % 6 + 1
            expected.append(value)
            replies.extend([{"question_id": question["id"], "answer": value, "confidence": 0.8,
                             "evidence": [{"field": "text", "quote": "Profile evidence"}]},
                            {"approve": True, "issues": []}])
        state = QuestionnaireAgent(ScriptedTransport(replies), BANK, self.tmp.name).run(
            "A", {"text": "Profile evidence"})
        exported = export_proxy(state, BANK)
        self.assertEqual(exported["item_responses"], expected)
        self.assertEqual(exported["answer_string"], "123456" * 10)
        self.assertEqual(len(validate_answers(exported["answer_string"], BANK)), 60)

    def test_notebook_score_mapping_and_question_order(self):
        payload = build_payload("123456" * 10, BANK)
        self.assertEqual([q["answer"] for q in payload["questions"]], [-3, -2, -1, 1, 2, 3] * 10)
        self.assertEqual([q["text"] for q in payload["questions"]], [q["text"] for q in BANK["questions"]])
        self.assertNotIn("subject_id", payload)
        self.assertEqual(ANSWER_SCORES, {1: -3, 2: -2, 3: -1, 4: 1, 5: 2, 6: 3})

    def test_invalid_or_incomplete_answers_are_rejected(self):
        for answers in ["1" * 59, "1" * 61, "7" * 60, "０" * 60,
                        [True] * 60, [None] * 60, [1.0] * 60, None]:
            with self.assertRaises(ValueError):
                validate_answers(answers, BANK)
        bank = copy.deepcopy(BANK); bank["questions"].reverse()
        with self.assertRaises(ValueError):
            build_payload("1" * 60, bank)

    def test_submit_uses_both_endpoints_and_validates_cache(self):
        crawler, session = self.crawler()
        session.cookies["XSRF-TOKEN"] = "session%20token"
        result = crawler.submit("1" * 60, BANK, "A", "self", self.tmp.name)
        self.assertEqual([c[0] for c in session.calls], ["GET", "POST", "POST"])
        self.assertTrue(session.calls[1][1].endswith("/test-results"))
        self.assertTrue(session.calls[2][1].endswith("/api/type-descriptions/results"))
        self.assertEqual(session.headers["X-XSRF-TOKEN"], "session token")
        self.assertEqual(session.calls[1][2]["json"], session.calls[2][2]["json"])
        self.assertIn("Energy: 65%", result["report_text"])
        self.assertIn("Independent & reflective.", result["report_text"])
        self.assertEqual(crawler.submit("1" * 60, BANK, "A", "self", self.tmp.name), result)
        self.assertEqual(len(session.calls), 3)
        with self.assertRaises(ValueError):
            crawler.submit("2" * 60, BANK, "A", "self", self.tmp.name)

    def test_http_forbidden_is_not_retried(self):
        crawler, session = self.crawler([Response(status=403)])
        with self.assertRaises(CrawlerError):
            crawler.submit("1" * 60, BANK, "A", "self", self.tmp.name)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(list(Path(self.tmp.name).glob("*")), [])

    def test_transient_status_has_bounded_retry(self):
        crawler, session = self.crawler([Response(status=429), Response(), Response({}), Response(REPORT)])
        crawler.submit("1" * 60, BANK, "A", "self", self.tmp.name)
        self.assertEqual(len(session.calls), 4)
        crawler, session = self.crawler([Response(status=503)] * 3)
        with self.assertRaises(CrawlerError):
            crawler.submit("1" * 60, BANK, "B", "self", self.tmp.name)
        self.assertEqual(len(session.calls), 3)

    def test_invalid_response_is_not_saved_as_a_report(self):
        for report in [Response(ValueError("not JSON")), Response({"message": "unexpected"})]:
            crawler, _ = self.crawler([Response(), Response({}), report])
            with self.assertRaises(CrawlerError):
                crawler.submit("1" * 60, BANK, "A", "self", self.tmp.name)
        self.assertEqual(list(Path(self.tmp.name).glob("*")), [])

    def test_agent_roles_preflight_and_use_report_texts(self):
        crawler, session = self.crawler([Response(), Response({}), Response(REPORT)] * 2)
        source = self.agent_output()
        result = attach_reports(source, BANK, crawler, self.tmp.name)
        self.assertEqual(set(result["questionnaire_reports"]), {"self", "partner"})
        self.assertEqual(result["semantic_texts"]["self_kernel"], render_report(REPORT))
        self.assertEqual(result["kernel_text_source"], "16personalities_report")
        self.assertEqual(source["semantic_texts"]["self_kernel"], "self items")
        self.assertEqual(len(session.calls), 6)
        self.assertEqual(session.calls[1][2]["json"]["questions"][0]["answer"], -2)
        self.assertEqual(session.calls[4][2]["json"]["questions"][0]["answer"], 2)

    def test_missing_answer_stops_both_roles_before_network(self):
        crawler, session = self.crawler()
        source = self.agent_output()
        source["kernels"]["partner"]["answer_string"] = None
        with self.assertRaises(ValueError):
            attach_reports(source, BANK, crawler, self.tmp.name)
        self.assertEqual(session.calls, [])

    def test_cli_dry_run_only_validates_sixty_digit_input(self):
        path = Path(self.tmp.name) / "answers.json"
        path.write_text(json.dumps({"A": "123456" * 10}))
        with patch("personality_crawler.PersonalityCrawler") as crawler:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["--answers-json", str(path)]), 0)
        crawler.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["items_per_agent"], 60)

    def test_report_html_cannot_insert_script_text(self):
        report = copy.deepcopy(REPORT)
        report["content"]["intro"]["parts"] = ["<script>ignore this</script><p>Visible</p>"]
        text = render_report(report)
        self.assertIn("Visible", text)
        self.assertNotIn("ignore this", text)

    def test_cognitive_cli_passes_sixty_answers_to_crawler(self):
        from cognitive_agent import main as cognitive_main
        profile_path = Path(self.tmp.name) / "profiles.json"
        profile_path.write_text(json.dumps({"A": {"text": "Profile context"}}))
        crawler, session = self.crawler([Response(), Response({}), Response(REPORT)] * 2)
        output = Path(self.tmp.name) / "agent_output"
        with patch("cognitive_agent.CognitiveAgent") as agent:
            agent.return_value.run.return_value = self.agent_output()
            with patch("personality_crawler.PersonalityCrawler", return_value=crawler):
                status = cognitive_main(["--profiles", str(profile_path), "--execute", "--crawl",
                                         "--output-dir", str(output), "--report-dir", self.tmp.name])
        self.assertEqual(status, 0)
        saved = read_json(next(output.glob("*.json")))
        self.assertEqual(saved["kernel_text_source"], "16personalities_report")
        self.assertIn("Personality Traits", saved["semantic_texts"]["partner_kernel"])
        self.assertEqual(len(session.calls), 6)

    def test_cognitive_cli_records_incomplete_answers_without_submission(self):
        from cognitive_agent import main as cognitive_main
        profile_path = Path(self.tmp.name) / "profiles.json"
        profile_path.write_text(json.dumps({"A": {"text": "Profile context"}}))
        source = self.agent_output()
        source["kernels"]["self"]["answer_string"] = None
        crawler, session = self.crawler()
        output = Path(self.tmp.name) / "agent_output"
        with patch("cognitive_agent.CognitiveAgent") as agent:
            agent.return_value.run.return_value = source
            with patch("personality_crawler.PersonalityCrawler", return_value=crawler):
                status = cognitive_main(["--profiles", str(profile_path), "--execute", "--crawl",
                                         "--output-dir", str(output), "--report-dir", self.tmp.name])
        self.assertEqual(status, 2)
        saved = read_json(next(output.glob("*.json")))
        self.assertEqual(saved["stage"], "questionnaire_report")
        self.assertEqual(saved["status"], "needs_review")
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
