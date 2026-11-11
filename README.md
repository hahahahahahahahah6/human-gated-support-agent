# human-gated-support-agent

## What it is

A Python 3.9+ support agent MVP with a handwritten state machine, local keyword retrieval, evidence citations, durable JSON checkpoints, and an explicit human approval gate. Version 0.1.0. Runtime dependencies: Python standard library only. No network access or API keys are used.

## How it works

```text
INTENT -> RETRIEVE -> DRAFT -> REVIEW
                               |-> EXECUTE -> DONE (safe)
                               |-> AWAITING_APPROVAL
                               |      |-> approve -> EXECUTE -> DONE
                               |      |-> reject  -> REFUSE  -> DONE
                               |-> REFUSE -> DONE (ungrounded)
                               |-> ESCALATE -> DONE (low confidence)
```

Intent rules cover refund, password reset, cancellation, shipping, billing, account closure, email changes, and general questions. Retrieval lowercases and tokenizes text, removes stopwords, and scores overlap divided by query token count plus a 0.2 matching-intent bonus. The top two documents are retained; the grounding threshold is 0.15. Drafts concatenate eligible excerpts verbatim. Ungrounded messages receive an evidence refusal and a suggestion to contact a human.

Checkpoints default to `.checkpoints/`. Each ticket has one JSON file, written through a temporary file and replacement. Ticket IDs allow letters, digits, underscores and hyphens. Existing tickets cannot be overwritten with `run`. `resume` and `status` reload and display the saved state without replaying an action; use `approve` or `reject` to continue a paused ticket. Malformed checkpoints and invalid gate operations exit 2. Refusals and pauses are successful workflow results and exit 0. Rejection records `outcome="REFUSED"` in terminal state `DONE`.

## Quickstart

```sh
python3 -m venv .venv
.venv/bin/python3 -m pip install .
.venv/bin/human-gated-support-agent --version
.venv/bin/python3 -m human_gated_support_agent run --ticket S1 --message "Where is shipment tracking?" --json
```

Source checkout usage needs no installation:

```sh
python3 -m human_gated_support_agent run --ticket S2 --message "How long is the warranty?"
python3 -m human_gated_support_agent status --ticket S2
python3 -m human_gated_support_agent resume --ticket S2
```

All ticket commands accept `--checkpoint-dir DIR`. `run --json` returns the state, answer, retrieved documents, citations, action, approval identity, execution timestamp, outcome and history. Approval, rejection, resume and status always print JSON.

## Human gate

Refund, cancellation, password reset, account closure and email change pause before execution. Safe answers and lookups execute directly. A second execution or approval is refused. The executor also raises `GateBlockedError` for direct dangerous execution without approval and writes no execution record.

```sh
human-gated-support-agent run --ticket T1 --message "Refund my unused product" --checkpoint-dir .checkpoints --json
human-gated-support-agent approve --ticket T1 --checkpoint-dir .checkpoints
human-gated-support-agent run --ticket T2 --message "Cancel my order" --checkpoint-dir .checkpoints
human-gated-support-agent reject --ticket T2 --checkpoint-dir .checkpoints --reason "Not authorized"
```

Approval records `approved_by="human:cli"`. `run --auto-approve` is for demos and evaluation only and records `human:auto-approve`. Execution records a UTC timestamp and the simulated action; it performs no external side effects.

## Evidence citations

The bundled KB contains eight documents covering refunds, shipping, password resets, cancellation, billing, account closure, email changes and warranty. Each grounded draft includes `citations: [{"doc_id": "...", "excerpt": "..."}]`; human output prints document IDs and excerpts too. Citations are copied from retrieved documents. Refusals have an empty citation list, so an unrelated retrieved document is never presented as evidence for a refused claim.

## Eval

Exactly 20 cases live in `eval/cases.jsonl`: 10 safe answer cases, 4 ungrounded cases and 6 dangerous cases. An identical package-data copy supports installed execution. The evaluator uses the real pipeline and fresh temporary checkpoints under the supplied directory. For each gate case it checks the initial pause, attempts direct unapproved execution, verifies blocking, and then approves and checks execution. Evaluation checkpoints are cleaned up afterward.

Exact command run from the source checkout:

```sh
python3 -m human_gated_support_agent eval --json --checkpoint-dir /tmp/hgsa-eval-checkpoints
```

Observed results (2026-11-10 PST):

| Metric | Observed value |
| --- | ---: |
| total | 20 |
| correct_intent (count) | 20 |
| grounded_accuracy (10 answer cases) | 1.0 |
| refusal_rate (4 refusal cases) | 1.0 |
| gate_block_rate (6 gate cases) | 1.0 |
| gate_execute_rate (6 gate cases) | 1.0 |
| forbidden_executions | 0 |
| hallucinated_citations | 0 |

These are measurements on a small, curated demo set, not evidence of general support accuracy. `eval --cases PATH` accepts another JSONL set; `eval` without `--json` prints metrics JSON followed by a human summary. Repeated runs produce identical metrics; timestamps are excluded from metrics.

## Verification

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile human_gated_support_agent/__init__.py human_gated_support_agent/__main__.py human_gated_support_agent/core.py human_gated_support_agent/cli.py human_gated_support_agent/eval.py tests/test_agent.py
python3 -m build --no-isolation
```

All 72 unittest tests passed and all sources compiled on Python 3.12. The suite covers each dangerous action through direct API and CLI paths, evidence fidelity, checkpoint round trips, illegal transitions, rejection, repeat execution, and deterministic evaluation. Python 3.9–3.11 were not available for runtime testing.

Both sdist and wheel were built with installed `build` 1.6.1 using `--no-isolation` to avoid network dependency downloads. A fresh venv initially lacked setuptools. Locally installed setuptools and wheel were copied into that venv as offline build tooling, then `python3 -m pip install --no-index --no-build-isolation .` succeeded there. Both installed entry points and the bundled eval were verified from outside the source checkout. No downloads were needed.

## Honest limitations

- Rule-based intent classification is brittle: negation, ambiguous requests and multiple intents can be misclassified. A policy question mentioning a dangerous intent is conservatively gated too.
- Keyword retrieval is not semantic search. A matching intent adds a bonus, so lexical overlap and citations do not prove that evidence answers the user's precise question.
- The bundled KB is a demo, not a real company policy. Answers consist of excerpts rather than personalized reasoning.
- Auto-approve is demo-only. Approval identities are labels, not authenticated users. The local API and checkpoints assume a trusted operator and writable local storage; malicious checkpoint edits are outside this MVP's trust boundary.
- No real payments or refunds are performed. Actions are recorded, not executed against any live system. Escalation is a recorded outcome and suggestion, not a message sent to a support team.
- Files are replaced atomically, but there is no multi-process locking, transactional database, tamper-evident audit log, or production durability guarantee. Use a single process per checkpoint directory.

MIT License, copyright 2026 Hao Li.
