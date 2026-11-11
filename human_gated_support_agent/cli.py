"""Command line interface; all actions are local simulations."""
import argparse
import json
import shlex
import sys
from . import __version__
from .core import Agent, CheckpointError, GateBlockedError
from .eval import run_eval

def main(argv=None):
    parser = argparse.ArgumentParser(prog='human-gated-support-agent')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'approve', 'reject', 'resume', 'status', 'eval'):
        command = sub.add_parser(name)
        command.add_argument('--checkpoint-dir', default='.checkpoints')
        command.add_argument('--json', action='store_true')
        if name != 'eval':
            command.add_argument('--ticket', required=True)
        if name == 'run':
            command.add_argument('--message', required=True)
            command.add_argument('--auto-approve', action='store_true', help='Demo only: simulate human approval')
        if name == 'reject':
            command.add_argument('--reason', default='Rejected by human')
        if name == 'eval':
            command.add_argument('--cases')
    args = parser.parse_args(argv)
    try:
        agent = Agent(args.checkpoint_dir)
        if args.command == 'eval':
            result = run_eval(args.cases, args.checkpoint_dir)
            print(json.dumps(result, sort_keys=True))
            if not args.json:
                print('Evaluated %d cases; grounded accuracy %.2f; gate block rate %.2f.' % (result['total'], result['grounded_accuracy'], result['gate_block_rate']))
            return 0
        if args.command == 'run':
            result = agent.run(args.message, args.ticket, args.auto_approve)
        elif args.command == 'approve':
            result = agent.approve(args.ticket)
        elif args.command == 'reject':
            result = agent.reject(args.ticket, args.reason)
        else:
            result = agent.resume(args.ticket)
        if args.command == 'run' and result['state'] == 'AWAITING_APPROVAL':
            result['approval_command'] = 'human-gated-support-agent approve --ticket %s --checkpoint-dir %s' % (shlex.quote(args.ticket), shlex.quote(args.checkpoint_dir))
        if args.json or args.command != 'run':
            print(json.dumps(result, sort_keys=True))
        else:
            print(result['answer'])
            for citation in result['citations']:
                print('[%s] %s' % (citation['doc_id'], citation['excerpt']))
            print('State: %s; outcome: %s' % (result['state'], result['outcome']))
            if 'approval_command' in result:
                print(result['approval_command'])
        return 0
    except (CheckpointError, GateBlockedError, ValueError, OSError) as exc:
        print('Error: %s' % exc, file=sys.stderr)
        return 2
