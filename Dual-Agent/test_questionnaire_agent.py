import json
import os
from pathlib import Path
import tempfile
import unittest
from questionnaire_agent import (QuestionnaireAgent, export_proxy, fingerprint,
                                  read_json, validate_answer, validate_bank)
from persona_agent import construct_persona


BANK = {"version": "synthetic-test", "questions": [
    {"id": 1, "text": "You regularly make new friends."},
    {"id": 2, "text": "Your workspace is organized."}]}
PROFILE = {"habits": "I regularly make new friends and keep my workspace organized."}


def answer(qid, rating=2, quote="I regularly make new friends"):
    return {"question_id": qid, "answer": rating, "confidence": 0.7,
            "evidence": [{"field": "habits", "quote": quote}]}


class ScriptedTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def __call__(self, messages):
        self.messages.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


OK = {"approve": True, "issues": []}


class QuestionnaireTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("ODREC_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)

    def agent(self, replies, bank=BANK, **kwargs):
        transport = ScriptedTransport(replies)
        return QuestionnaireAgent(transport, bank, self.tmp.name, **kwargs), transport

    def test_every_call_pins_the_same_profile_and_current_item(self):
        agent, transport = self.agent([answer(1), OK, answer(2, quote="workspace organized"), OK])
        state = agent.run("A", PROFILE)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(transport.messages), 4)
        for messages in transport.messages:
            self.assertIn(PROFILE["habits"], messages[0]["content"])
            self.assertIn(fingerprint(PROFILE), messages[0]["content"])
        self.assertEqual(json.loads(transport.messages[2][1]["content"])["current_question"]["id"], 2)
        self.assertEqual(len(json.loads(transport.messages[2][1]["content"])["recent_accepted_answers"]), 1)

    def test_reflector_rejection_revises_with_feedback(self):
        agent, transport = self.agent([answer(1), {"approve": False, "issues": ["answer too extreme"]},
            answer(1, 3), OK], bank={"questions": [BANK["questions"][0]]})
        state = agent.run("A", PROFILE)
        self.assertEqual(state["answers"]["1"]["answer"]["answer"], 3)
        self.assertIn("answer too extreme", transport.messages[2][1]["content"])
        self.assertEqual(len(state["attempts"]), 2)

    def test_bad_schema_and_fabricated_evidence_are_not_accepted(self):
        invalid = answer(1, True)
        agent, transport = self.agent(["not json", invalid, answer(1, quote="invented quote")],
            bank={"questions": [BANK["questions"][0]]})
        state = agent.run("A", PROFILE)
        self.assertEqual(state["status"], "needs_review")
        self.assertEqual(state["answers"], {})
        self.assertEqual(len(transport.messages), 3)
        with self.assertRaises(ValueError):
            export_proxy(state, BANK)

    def test_abstention_export(self):
        null = {"question_id": 1, "answer": None, "confidence": 0, "evidence": []}
        bank = {"questions": [BANK["questions"][0]]}
        agent, _ = self.agent([null, OK], bank=bank)
        proxy = export_proxy(agent.run("A", PROFILE), bank)
        self.assertEqual(proxy["answer_string"], "0")
        self.assertEqual(proxy["submission_responses"], [0])
        self.assertEqual(proxy["item_responses"], [None])
        self.assertIn("insufficient profile evidence", proxy["proxy_text"])

    def test_completed_session_resumes_without_api_calls(self):
        agent, _ = self.agent([answer(1), OK, answer(2), OK])
        agent.run("A", PROFILE)
        resumed, transport = self.agent([])
        self.assertEqual(resumed.run("A", PROFILE)["status"], "complete")
        self.assertFalse(transport.messages)

    def test_resume_preserves_accepted_items_after_interruption(self):
        agent, _ = self.agent([answer(1), OK, RuntimeError("offline")])
        with self.assertRaises(RuntimeError):
            agent.run("A", PROFILE)
        resumed, transport = self.agent([answer(2), OK])
        self.assertEqual(resumed.run("A", PROFILE)["status"], "complete")
        self.assertEqual(len(transport.messages), 2)
        self.assertEqual(json.loads(transport.messages[0][1]["content"])["current_question"]["id"], 2)

    def test_profile_or_settings_change_cannot_reuse_a_session(self):
        agent, _ = self.agent([answer(1), OK, answer(2), OK])
        agent.run("A", PROFILE)
        with self.assertRaises(ValueError):
            agent.run("A", {"habits": "different"})
        changed, _ = self.agent([], model_settings={"model": "different"})
        with self.assertRaises(ValueError):
            changed.run("A", PROFILE)

    def test_users_and_agent_roles_have_separate_external_memory(self):
        agent, transport = self.agent([answer(1), OK, answer(2), OK] * 3)
        for sid, role in [("../../A", "self"), ("B", "self"), ("../../A", "partner")]:
            agent.run(sid, PROFILE, role)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.json"))), 3)
        self.assertEqual(len(transport.messages), 12)

    def test_duplicate_ids_bad_ranges_and_nonfinite_confidence(self):
        with self.assertRaises(ValueError):
            validate_bank({"questions": [BANK["questions"][0]] * 2})
        bad = answer(1, 7)
        self.assertTrue(validate_answer(bad, BANK["questions"][0], PROFILE))
        bad = answer(1); bad["confidence"] = float("nan")
        self.assertTrue(validate_answer(bad, BANK["questions"][0], PROFILE))

    def test_reviewer_string_boolean_is_rejected(self):
        agent, _ = self.agent([answer(1), {"approve": "true", "issues": []}],
            bank={"questions": [BANK["questions"][0]]}, max_rounds=1)
        self.assertEqual(agent.run("A", PROFILE)["status"], "needs_review")

    def test_corrupted_cache_is_rejected(self):
        agent, _ = self.agent([answer(1), OK, answer(2), OK])
        agent.run("A", PROFILE)
        path = next(Path(self.tmp.name).glob("*.json"))
        state = read_json(path); state["answers"]["1"]["answer"]["answer"] = 9
        path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(ValueError):
            agent.run("A", PROFILE)


    def test_questionnaire_has_sixty_unique_items(self):
        bank = read_json(Path(__file__).parent / "questionnaire.json")
        validate_bank(bank)
        self.assertEqual(len(bank["questions"]), 60)

    def test_persona_preserves_observed_self_fields(self):
        prompts = read_json(Path(__file__).parent / "prompts.json")
        transport = ScriptedTransport([
            {"profile": {"habits": "changed"}, "inferred_fields": [], "evidence": {}},
            {"profile": PROFILE, "inferred_fields": [], "evidence": {}}, OK])
        result = construct_persona(PROFILE, "self", transport, prompts)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["persona"]["profile"], PROFILE)
        self.assertEqual(len(result["attempts"]), 2)

    def test_partner_proposal_requires_evidence_and_reflection(self):
        prompts = read_json(Path(__file__).parent / "prompts.json")
        proposal = {"profile": {"habits": "Enjoys meeting new people"},
            "inferred_fields": ["habits"], "evidence": {"habits": [
                {"field": "habits", "quote": "I regularly make new friends"}]}}
        transport = ScriptedTransport([proposal, OK])
        self.assertEqual(construct_persona(PROFILE, "partner", transport, prompts)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
