import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from human_gated_support_agent.core import (Agent, CheckpointError, GateBlockedError,
    ACTIONS, classify, draft, load_kb, retrieve, transition)
from human_gated_support_agent.cli import main
from human_gated_support_agent.eval import run_eval

MESSAGES = {'refund': 'Refund my unused product', 'password_reset': 'Reset my password',
            'cancel_order': 'Cancel my order', 'close_account': 'Close my account',
            'change_email': 'Change my email'}

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agent = Agent(self.temp.name)

    def cli(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(list(args) + ['--checkpoint-dir', self.temp.name])
        return code, output.getvalue(), error.getvalue()

    def test_unknown(self):
        self.assertEqual(classify('quantum zebra'), ('general', 0.1))

    def test_empty(self):
        self.assertEqual(classify(''), ('general', 0.1))

    def test_shipping_intent(self):
        self.assertEqual(classify('shipment tracking')[0], 'shipping_status')

    def test_billing_intent(self):
        self.assertEqual(classify('charge dispute invoice')[0], 'billing_question')

    def test_general_intent(self):
        self.assertEqual(classify('warranty')[0], 'general')

    def test_kb_size(self):
        self.assertGreaterEqual(len(load_kb()), 8)

    def test_kb_unique(self):
        docs = load_kb()
        self.assertEqual(len({d['id'] for d in docs}), len(docs))

    def test_kb_schema(self):
        for doc in load_kb():
            self.assertEqual(set(doc), {'id', 'title', 'intent', 'text'})
            self.assertGreaterEqual(len(doc['text'].split('. ')), 2)

    def test_threshold(self):
        self.assertFalse(retrieve('quasar zebra', 'general')[1])

    def test_empty_retrieval(self):
        self.assertFalse(retrieve('', 'general')[1])

    def test_top_two(self):
        self.assertEqual(len(retrieve('shipping', 'shipping_status')[0]), 2)

    def test_custom_threshold(self):
        self.assertFalse(retrieve('shipping', 'shipping_status', 10)[1])

    def test_refusal_draft(self):
        result = draft(retrieve('zebra', 'general')[0], False)
        self.assertEqual(result['citations'], [])
        self.assertIn('evidence', result['answer'])
        self.assertIn('human', result['answer'])
        self.assertFalse(any(c.isdigit() for c in result['answer']))

    def test_safe_execution(self):
        state = self.agent.run('Where is shipment tracking?', 'safe')
        self.assertEqual(state['executed_action'], 'lookup')
        self.assertIsNone(state['approved_by'])
        self.assertEqual(state['state'], 'DONE')

    def test_ungrounded(self):
        state = self.agent.run('Mars weather', 'unknown')
        self.assertEqual(state['outcome'], 'REFUSED')
        self.assertFalse(state['citations'])
        self.assertIsNone(state['executed_action'])

    def test_low_confidence_escalates(self):
        state = self.agent.run('unused products', 'low')
        self.assertEqual(state['outcome'], 'ESCALATED')
        self.assertIn('ESCALATE', state['history'])

    def test_roundtrip(self):
        state = self.agent.run('Refund please', 'roundtrip')
        self.assertEqual(state, self.agent.resume('roundtrip'))

    def test_duplicate_ticket(self):
        self.agent.run('Refund', 'duplicate')
        with self.assertRaises(CheckpointError):
            self.agent.run('shipping', 'duplicate')

    def test_missing_checkpoint(self):
        with self.assertRaises(CheckpointError):
            self.agent.resume('missing')

    def test_corrupt_checkpoint(self):
        Path(self.temp.name, 'bad.json').write_text('{')
        with self.assertRaises(CheckpointError):
            self.agent.resume('bad')

    def test_incomplete_checkpoint(self):
        Path(self.temp.name, 'bad.json').write_text('{}')
        with self.assertRaises(CheckpointError):
            self.agent.resume('bad')

    def test_invalid_ticket(self):
        with self.assertRaises(CheckpointError):
            self.agent.run('refund', '../escape')

    def test_illegal_transition(self):
        with self.assertRaises(ValueError):
            transition({'state': 'INTENT', 'history': ['INTENT']}, 'EXECUTE')

    def test_terminal_transition(self):
        with self.assertRaises(ValueError):
            transition({'state': 'DONE', 'history': ['DONE']}, 'INTENT')

    def test_legal_transition(self):
        state = dict(state='INTENT', history=['INTENT'])
        transition(state, 'RETRIEVE')
        self.assertEqual(state['history'], ['INTENT', 'RETRIEVE'])

    def test_auto_approve(self):
        state = self.agent.run('Refund', 'auto', True)
        self.assertEqual(state['approved_by'], 'human:auto-approve')
        self.assertEqual(state['executed_action'], 'refund')

    def test_blank_approver(self):
        self.agent.run('Refund', 'blank')
        with self.assertRaises(GateBlockedError):
            self.agent.approve('blank', ' ')

    def test_cli_corrupt(self):
        Path(self.temp.name, 'bad.json').write_text('{')
        code, _, error = self.cli('resume', '--ticket', 'bad')
        self.assertEqual(code, 2)
        self.assertIn('checkpoint', error)

    def test_cli_status_readonly(self):
        state = self.agent.run('refund', 'status')
        before = self.agent.path('status').read_bytes()
        code, output, _ = self.cli('status', '--ticket', 'status')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output), state)
        self.assertEqual(self.agent.path('status').read_bytes(), before)

    def test_cli_resume(self):
        state = self.agent.run('refund', 'resume')
        code, output, _ = self.cli('resume', '--ticket', 'resume')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output), state)

    def test_cli_approve(self):
        self.agent.run('refund', 'approve')
        code, output, _ = self.cli('approve', '--ticket', 'approve')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)['executed_action'], 'refund')

    def test_cli_reject(self):
        self.agent.run('refund', 'reject')
        code, output, _ = self.cli('reject', '--ticket', 'reject', '--reason', 'No')
        self.assertEqual(code, 0)
        state = json.loads(output)
        self.assertEqual(state['outcome'], 'REFUSED')
        self.assertEqual(state['rejection_reason'], 'No')
        self.assertIsNone(state['executed_action'])

    def test_cli_safe_json(self):
        code, output, _ = self.cli('run', '--message', 'warranty', '--ticket', 'safe', '--json')
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output)['citations'])

    def test_cli_human(self):
        code, output, _ = self.cli('run', '--message', 'refund', '--ticket', 'human')
        self.assertEqual(code, 0)
        self.assertIn('approve --ticket human', output)

    def test_cli_eval(self):
        code, output, _ = self.cli('eval', '--json')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)['total'], 20)

    def test_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(['--version'])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), '0.1.0')

    def test_eval_determinism(self):
        self.assertEqual(run_eval(), run_eval())

    def test_eval_metrics(self):
        metrics = run_eval()
        self.assertEqual(metrics['total'], 20)
        self.assertEqual(metrics['correct_intent'], 20)
        for key in ('grounded_accuracy', 'refusal_rate', 'gate_block_rate', 'gate_execute_rate'):
            self.assertEqual(metrics[key], 1.0)
        self.assertEqual(metrics['forbidden_executions'], 0)
        self.assertEqual(metrics['hallucinated_citations'], 0)

    def test_eval_distribution(self):
        cases = [json.loads(line) for line in Path('eval/cases.jsonl').read_text().splitlines()]
        self.assertEqual([sum(c['expects'] == kind for c in cases) for kind in ('answer', 'refuse', 'gate')], [10, 4, 6])
        self.assertEqual(len({c['id'] for c in cases}), 20)
        self.assertEqual(Path('eval/cases.jsonl').read_bytes(), Path('human_gated_support_agent/cases.jsonl').read_bytes())

# Each generated method is an independent test of one intent/action/doc invariant.
def intent_test(intent, message):
    def test(self): self.assertEqual(classify(message)[0], intent)
    return test

def gate_test(intent, message):
    def test(self):
        state = self.agent.run(message, 'gate')
        self.assertEqual(state['state'], 'AWAITING_APPROVAL')
        self.assertIsNone(state['executed_action'])
        attempt = copy.deepcopy(state)
        transition(attempt, 'EXECUTE')
        before = self.agent.path('gate').read_bytes()
        with self.assertRaises(GateBlockedError): self.agent.execute(attempt)
        self.assertIsNone(attempt['executed_action'])
        self.assertEqual(before, self.agent.path('gate').read_bytes())
    return test

def approve_test(intent, message):
    def test(self):
        self.agent.run(message, 'approve')
        state = self.agent.approve('approve', 'human:tester')
        self.assertEqual(state['executed_action'], ACTIONS[intent])
        self.assertEqual(state['approved_by'], 'human:tester')
        self.assertTrue(state['timestamp'])
        with self.assertRaises(GateBlockedError): self.agent.execute(state)
        with self.assertRaises(GateBlockedError): self.agent.approve('approve')
    return test

def reject_test(intent, message):
    def test(self):
        self.agent.run(message, 'reject')
        state = self.agent.reject('reject', 'Not authorized')
        self.assertEqual(state['outcome'], 'REFUSED')
        self.assertIsNone(state['executed_action'])
        with self.assertRaises(GateBlockedError): self.agent.approve('reject')
    return test

def cli_gate_test(intent, message):
    def test(self):
        code, output, _ = self.cli('run', '--message', message, '--ticket', 'gate', '--json')
        self.assertEqual(code, 0)
        state = json.loads(output)
        self.assertEqual(state['state'], 'AWAITING_APPROVAL')
        self.assertIsNone(state['executed_action'])
        self.assertIn('approve', state['approval_command'])
    return test

for intent, message in MESSAGES.items():
    for name, factory in [('intent', intent_test), ('gate', gate_test), ('approve', approve_test), ('reject', reject_test), ('cli_gate', cli_gate_test)]:
        setattr(WorkflowTests, 'test_%s_%s' % (name, intent), factory(intent, message))

def retrieval_test(doc):
    def test(self):
        retrieved, grounded = retrieve(doc['title'], doc['intent'])
        self.assertTrue(grounded)
        self.assertEqual(retrieved[0]['doc_id'], doc['id'])
        result = draft(retrieved, grounded)
        evidence = {(d['doc_id'], d['excerpt']) for d in retrieved}
        self.assertTrue(result['citations'])
        for citation in result['citations']:
            self.assertIn((citation['doc_id'], citation['excerpt']), evidence)
        self.assertEqual(result['answer'], ' '.join(c['excerpt'] for c in result['citations']))
    return test
for doc in load_kb():
    setattr(WorkflowTests, 'test_retrieval_' + doc['id'], retrieval_test(doc))
