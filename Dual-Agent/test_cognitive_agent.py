import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from cognitive_agent import CognitiveAgent, main
from persona_agent import construct_persona
from questionnaire_agent import (QuestionnaireAgent, export_proxy, read_json,
    validate_bank, normalize_profile)
from test_questionnaire_agent import ScriptedTransport, OK

BANK = {"questions": [{"id": 1, "text": "You enjoy quiet routines."}]}
SOURCE = {"habits": "I enjoy quiet routines."}
PARTNER = {"habits": "Prefers a quiet daily routine."}
SELF_PERSONA = {"profile": SOURCE, "inferred_fields": [], "evidence": {}}
PARTNER_PERSONA = {"profile": PARTNER, "inferred_fields": ["habits"],
    "evidence": {"habits": [{"field": "habits", "quote": SOURCE["habits"]}]}}


def response(profile):
    return {"question_id": 1, "answer": 2, "confidence": 0.7,
            "evidence": [{"field": "habits", "quote": profile["habits"]}]}


class CognitiveAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('ODREC_TEST_TMPDIR'))
        self.addCleanup(self.tmp.cleanup)

    def build(self, replies):
        transport = ScriptedTransport(replies)
        return CognitiveAgent(transport, BANK, self.tmp.name), transport

    def replies(self):
        return [SELF_PERSONA, OK, response(SOURCE), OK,
                PARTNER_PERSONA, OK, response(PARTNER), OK]

    def test_self_uses_source_partner_uses_generated_persona(self):
        agent, transport = self.build(self.replies())
        result = agent.run('A', SOURCE)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(set(result['semantic_texts']),
            {'self_persona', 'self_kernel', 'partner_persona', 'partner_kernel'})
        self.assertIn(SOURCE['habits'], transport.messages[2][0]['content'])
        self.assertIn(PARTNER['habits'], transport.messages[6][0]['content'])
        self.assertNotIn(SOURCE['habits'], transport.messages[6][0]['content'])
        self.assertEqual(len(transport.messages), 8)

    def test_resume_reuses_personas_and_completed_questionnaires(self):
        agent, _ = self.build(self.replies())
        original = agent.run('A', SOURCE)
        resumed, transport = self.build([])
        self.assertEqual(resumed.run('A', SOURCE), original)
        self.assertEqual(transport.messages, [])

    def test_rejected_persona_never_enters_questionnaire(self):
        agent, transport = self.build([SELF_PERSONA, {'approve': False, 'issues': ['unsupported']}]*3)
        result = agent.run('A', SOURCE)
        self.assertEqual(result['stage'], 'self_persona')
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(len(transport.messages), 6)
        self.assertFalse(list((Path(self.tmp.name)/'questionnaire').glob('*.json')))

    def test_changed_source_does_not_reuse_persona_memory(self):
        agent, _ = self.build(self.replies())
        agent.run('A', SOURCE)
        with self.assertRaises(ValueError):
            agent.run('A', {'habits': 'Different source'})

    def test_persona_cache_is_validated_before_use(self):
        agent, _ = self.build(self.replies())
        agent.run('A', SOURCE)
        paths = list((Path(self.tmp.name)/'personas').glob('*.json'))
        path = next(p for p in paths if read_json(p)['provenance']['role'] == 'self')
        data = read_json(path);data['result']['persona']['profile']['habits'] = 'Tampered'
        path.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaises(ValueError):
            agent.run('A', SOURCE)

    def test_malformed_persona_fields_are_rejected_without_crashing(self):
        prompts = read_json(Path(__file__).parent/'prompts.json')
        transport = ScriptedTransport([{'profile': PARTNER, 'inferred_fields': [[]], 'evidence': {}}])
        result = construct_persona(SOURCE, 'partner', transport, prompts, max_rounds=1)
        self.assertEqual(result['status'], 'needs_review')

    def test_nonprefix_or_malformed_cached_records_raise_value_error(self):
        for malformed in ({'99': {}}, {'1': []}):
            with tempfile.TemporaryDirectory(dir=self.tmp.name) as memory:
                transport = ScriptedTransport([response(SOURCE), OK])
                agent = QuestionnaireAgent(transport, BANK, memory)
                agent.run('A', SOURCE)
                path = next(Path(memory).glob('*.json'))
                state = read_json(path);state['answers'] = malformed
                path.write_text(json.dumps(state), encoding='utf-8')
                with self.assertRaises(ValueError):
                    agent.run('A', SOURCE)

    def test_export_rejects_wrong_questionnaire_and_tampered_answers(self):
        agent = QuestionnaireAgent(ScriptedTransport([response(SOURCE), OK]), BANK, self.tmp.name)
        state = agent.run('A', SOURCE)
        with self.assertRaises(ValueError):
            export_proxy(state, {'questions': [{'id': 1, 'text': 'Changed'}]})
        state['answers']['1']['answer']['answer'] = 9
        with self.assertRaises(ValueError):
            export_proxy(state, BANK)

    def test_invalid_bank_and_profile_keys_raise_value_error(self):
        for bank in ([], {'questions': None}, {'questions': ['invalid']}):
            with self.assertRaises(ValueError):
                validate_bank(bank)
        with self.assertRaises(ValueError):
            normalize_profile({1: 'a', '1': 'b'})

    def test_dry_run_creates_no_memory_and_calls_no_api(self):
        profiles = Path(self.tmp.name)/'profiles.json'
        profiles.write_text(json.dumps({'A': SOURCE}), encoding='utf-8')
        memory = Path(self.tmp.name)/'not-created'
        with mock.patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--profiles', str(profiles), '--memory-dir', str(memory)]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['agent_roles'], ['self', 'partner'])
        self.assertEqual(result['maximum_logical_model_calls'], 3*(4+4*60))
        self.assertFalse(memory.exists())


if __name__ == '__main__':
    unittest.main()
