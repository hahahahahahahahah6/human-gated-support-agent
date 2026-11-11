"""Deterministic support workflow. Actions are local audit records only."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from importlib.resources import files

class GateBlockedError(ValueError):
    """Execution was attempted without approval or in an invalid state."""

class CheckpointError(ValueError):
    """Missing, malformed, or incompatible checkpoint."""

RULES = [
    ('refund', ('refund',)),
    ('password_reset', ('password', 'reset link')),
    ('cancel_order', ('cancel', 'cancellation')),
    ('shipping_status', ('shipping', 'shipment', 'tracking', 'delivery')),
    ('billing_question', ('billing', 'invoice', 'charge', 'dispute')),
    ('close_account', ('close account', 'close my account', 'account closure', 'delete account')),
    ('change_email', ('change email', 'change my email', 'email change', 'new address')),
]
ACTIONS = dict(refund='refund', password_reset='reset_password', cancel_order='cancel_order', close_account='close_account', change_email='change_email')
DANGEROUS = frozenset(ACTIONS.values())
TRANSITIONS = {'INTENT': {'RETRIEVE'}, 'RETRIEVE': {'DRAFT'}, 'DRAFT': {'REVIEW'}, 'REVIEW': {'EXECUTE', 'AWAITING_APPROVAL', 'REFUSE', 'ESCALATE'}, 'AWAITING_APPROVAL': {'EXECUTE', 'REFUSE'}, 'EXECUTE': {'DONE'}, 'REFUSE': {'DONE'}, 'ESCALATE': {'DONE'}, 'DONE': set()}
STOP = set('a an the is are to for of my i me please what how can do you it in and within only about tell your'.split())

def tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - STOP

def classify(message):
    text = message.lower()
    for intent, phrases in RULES:
        if any(phrase in text for phrase in phrases):
            return intent, 0.9
    return 'general', 0.8 if 'warranty' in text else 0.1

def load_kb():
    return json.loads(files('human_gated_support_agent').joinpath('kb.json').read_text())

def retrieve(message, intent, threshold=0.15):
    query = tokens(message)
    ranked = []
    for doc in load_kb():
        overlap = len(query & tokens(doc['text'] + ' ' + doc['title'])) / max(len(query), 1)
        score = overlap + (0.2 if query and intent != 'general' and intent == doc['intent'] else 0)
        ranked.append(dict(doc_id=doc['id'], title=doc['title'], excerpt=doc['text'], score=score))
    ranked.sort(key=lambda d: (-d['score'], d['doc_id']))
    return ranked[:2], ranked[0]['score'] >= threshold

def draft(retrieved, grounded):
    if not grounded:
        return {'answer': 'I cannot answer from the available evidence. Please escalate to a human.', 'citations': []}
    evidence = [doc for doc in retrieved if doc['score'] >= 0.15]
    return {'answer': ' '.join(doc['excerpt'] for doc in evidence), 'citations': [dict(doc_id=doc['doc_id'], excerpt=doc['excerpt']) for doc in evidence]}

def transition(state, target):
    if target not in TRANSITIONS.get(state['state'], set()):
        raise ValueError('Illegal transition: %s -> %s' % (state['state'], target))
    state['state'] = target
    state['history'].append(target)

class Agent:
    def __init__(self, checkpoint_dir='.checkpoints'):
        self.directory = Path(checkpoint_dir)

    def path(self, ticket):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', ticket):
            raise CheckpointError('Ticket must contain only letters, digits, underscore or hyphen')
        return self.directory / (ticket + '.json')

    def save(self, state):
        path = self.path(state['ticket'])
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(path)

    def resume(self, ticket):
        try:
            state = json.loads(self.path(ticket).read_text())
            required = {'ticket', 'state', 'history', 'intent', 'action', 'approved_by', 'executed_action', 'retrieved', 'citations', 'answer', 'outcome', 'grounded', 'confidence', 'message', 'timestamp'}
            if not isinstance(state, dict) or not required <= state.keys():
                raise ValueError('missing fields')
            if state['ticket'] != ticket or state['state'] not in TRANSITIONS or not isinstance(state['history'], list) or state['history'][-1:] != [state['state']]:
                raise ValueError('invalid state')
            if state['action'] not in DANGEROUS | {'answer_only', 'lookup'}:
                raise ValueError('invalid action')
            if state['executed_action'] and state['action'] in DANGEROUS and not state['approved_by']:
                raise ValueError('unapproved execution record')
            return state
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise CheckpointError('Cannot load checkpoint for %s: %s' % (ticket, exc)) from exc

    def execute(self, state):
        if state['state'] != 'EXECUTE' or state.get('executed_action'):
            raise GateBlockedError('Execution requires EXECUTE state and cannot be repeated')
        if state['action'] in DANGEROUS and not state.get('approved_by'):
            raise GateBlockedError('Human approval required')
        state['executed_action'] = state['action']
        state['timestamp'] = datetime.now(timezone.utc).isoformat()
        state['outcome'] = 'EXECUTED'
        transition(state, 'DONE')
        self.save(state)
        return state

    def run(self, message, ticket, auto_approve=False):
        if self.path(ticket).exists():
            raise CheckpointError('Ticket already exists; use resume, approve or reject')
        intent, confidence = classify(message)
        state = dict(ticket=ticket, message=message, state='INTENT', history=['INTENT'], intent=intent, confidence=confidence, action=ACTIONS.get(intent, 'lookup' if intent == 'shipping_status' else 'answer_only'), approved_by=None, executed_action=None, timestamp=None, outcome=None)
        transition(state, 'RETRIEVE')
        state['retrieved'], state['grounded'] = retrieve(message, intent)
        transition(state, 'DRAFT')
        state.update(draft(state['retrieved'], state['grounded']))
        transition(state, 'REVIEW')
        if not state['grounded'] or confidence < 0.5:
            transition(state, 'REFUSE' if not state['grounded'] else 'ESCALATE')
            state['outcome'] = 'REFUSED' if not state['grounded'] else 'ESCALATED'
            transition(state, 'DONE')
            self.save(state)
            return state
        if state['action'] in DANGEROUS:
            transition(state, 'AWAITING_APPROVAL')
            self.save(state)
            return self.approve(ticket, 'human:auto-approve') if auto_approve else state
        transition(state, 'EXECUTE')
        return self.execute(state)

    def approve(self, ticket, approved_by='human:cli'):
        state = self.resume(ticket)
        if state['state'] != 'AWAITING_APPROVAL' or not approved_by.strip():
            raise GateBlockedError('Only a paused ticket can be approved by a named human')
        state['approved_by'] = approved_by
        transition(state, 'EXECUTE')
        return self.execute(state)

    def reject(self, ticket, reason='Rejected by human'):
        state = self.resume(ticket)
        if state['state'] != 'AWAITING_APPROVAL':
            raise GateBlockedError('Only a paused ticket can be rejected')
        transition(state, 'REFUSE')
        state.update(outcome='REFUSED', rejection_reason=reason)
        transition(state, 'DONE')
        self.save(state)
        return state
