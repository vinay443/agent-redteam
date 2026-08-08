# agent-redteam

A red-teaming lab for tool-using LLM agents. An autonomous **target agent** with
real file-system tools runs inside a locked-down Docker container while a
second, LLM-driven **attacker** tries to break it — via prompt injection, goal
hijacking, permission escalation, and system-prompt exfiltration. A **judge**
scores every attempt (by code where possible, by an LLM where not), and a
**report** turns the results into a per-category success-rate table and a
Markdown write-up.

The point is measurement: *how often, and by which layer, is a tool-using agent
actually stopped?* — and the harness is built so that answer is never taken on
faith. Every attack, tool call, and verdict is logged with a timestamp, and the
containment that makes the whole thing safe is [enforced in code and
independently verifiable](SAFETY.md).

> ⚠️ **Read [SAFETY.md](SAFETY.md) before running.** This tool deliberately
> induces an agent to misbehave. It is safe only because the agent is boxed;
> SAFETY.md documents the box and how to verify it.

---

## What it measures

Four attack categories, each with a clear definition of "the attacker won":

| Category | Question | How it's judged |
| --- | --- | --- |
| `prompt_injection` | Does content read from a file override the agent's instructions? | **Code** — each attack names a concrete artefact the agent produces only if it obeyed the injection. |
| `goal_hijacking` | Does the agent abandon the user's task for the attacker's? | **LLM** — a rubric-driven judge decides whether the real task was dropped. |
| `permission_escalation` | Can the agent be made to touch anything outside its workspace? | **Code** — success requires a file op that *escaped* the sandbox root. Should always be 0%. |
| `system_prompt_exfiltration` | Can the agent be made to reveal its instructions? | **Code + LLM** — an exact per-run canary-token check, plus an LLM looking for paraphrased leaks. |

Two secondary signals are reported alongside the headline success rate:

- **Blocked-by-code share** — of the attacks that failed, how many were stopped
  by a code boundary versus by the model choosing to decline. This separates
  *structural* safety (holds regardless of the model) from *behavioural* safety.
- **Containment integrity** — the count of tool calls that escaped the sandbox,
  which must be zero. A non-zero value is a bug, flagged in red.

---

## Architecture

```
                        ┌───────────────────────────────────────────────┐
  attacker/             │  runner/  (orchestrator, on the host)          │
  ├ seeds.yaml          │                                                │
  ├ generator.py  ──────┼──► build campaign ─► for each attack:          │
  │  (LLM variants)     │      provision ./sandbox/runs/<run>/<attack>/  │
  └ campaign.py         │      dispatch ──────────────┐                  │
                        │                             ▼                  │
                        │              ┌──────────────────────────────┐  │
                        │              │  Docker container (target)   │  │
                        │  docker exec │  target_agent/               │  │
                        │  -i  ───────►│   agent.py  (tool-use loop)  │  │
                        │              │   tools.py  ─► sandbox.py    │  │
                        │              │   guard.py  (defence seam)   │  │
                        │              │  mount: ./sandbox only       │  │
                        │              │  512m / 1cpu / pids=128 / RO │  │
                        │              └──────────────────────────────┘  │
                        │                             │ AgentRun (JSON)   │
                        │      judge/ ◄───────────────┘                   │
                        │      ├ signals.py    (objective facts)          │
                        │      ├ code_checks.py (success_signal)          │
                        │      └ llm_judge.py  (rubric verdict)           │
                        │                             │                   │
                        │      runner/store.py ◄──────┘                   │
                        │      SQLite + JSONL mirror + events.jsonl       │
                        └───────────────────────────────────────────────┘
                                            │
                          report/  ─► table + report.md
```

- **`common/`** — config, timestamped JSONL logging, id generation, the checked
  Anthropic client wrapper.
- **`target_agent/`** — the agent under test: a raw function-calling loop
  (`agent.py`) over two sandboxed tools (`tools.py` + `sandbox.py`), a system
  prompt with a per-run canary (`prompts.py`), the egress allowlist (`net.py`),
  and the `guard.py` seam where a defence layer slots in later. **This package
  is the only application code in the container image** — the target cannot
  import its own attacker or judge.
- **`attacker/`** — seed corpus (`seeds.yaml`), an LLM variant generator, and
  campaign assembly.
- **`judge/`** — objective signal extraction, code-based success checks, and an
  LLM-as-judge with explicit rubrics.
- **`runner/`** — container lifecycle, per-attack dispatch, and the SQLite/JSONL
  store.
- **`report/`** — metric aggregation and rendering.
- **`docker/`** — the target image and its resource-limited compose config.
- **`tests/`** — the safety-critical surface, path-traversal check included.

---

## Quickstart

### 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

### 3. Verify containment (recommended before the first run)

```bash
pytest tests/test_sandbox.py -v          # host-side path-traversal check
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker exec -i agent-redteam-target python -m target_agent --selftest
docker compose -f docker/docker-compose.yml down -v
```

### 4. Run a campaign

```bash
# 10 prompt-injection attacks (seeds + LLM-generated variants), containerised:
python -m runner --category prompt_injection --n 10

# All four categories, 5 each:
python -m runner --category all --n 5

# Require Docker (fail if unavailable) — the right flag for real measurement:
python -m runner --category permission_escalation --n 8 --docker
```

### 5. Report

```bash
python -m report                    # latest run: table + results/<run>/report.md
python -m report --run-id run-...    # a specific run
python -m report --list             # available runs
python -m report --json             # metrics as JSON
```

---

## CLI reference

### `python -m runner`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--category` | `all` | `prompt_injection` \| `goal_hijacking` \| `permission_escalation` \| `system_prompt_exfiltration` \| `all` |
| `--n` | `5` | Attacks per category (seed + generated variants) |
| `--guard` | `none` | Defence layer to apply to the target (see Extending) |
| `--docker` / `--no-docker` | auto | Require / forbid containerised execution |
| `--no-variants` | off | Skip the attacker model; repeat seeds to fill `--n` |
| `--no-llm-judge` | off | Disable the LLM judge (LLM-only attacks scored as failures) |
| `--no-seed` | off | Run generated variants only, excluding the hand-written seed |
| `--run-id` | generated | Override the run id |

Auto executor selection: if Docker is available, the runner builds the image,
starts the container, runs the in-container self-test, and only proceeds if it
passes. Otherwise it falls back to **in-process execution**, clearly tagged
`sandboxed=False` — fine for development, not for real measurement. Use
`--docker` to require the sandbox.

### `python -m report`

| Flag | Meaning |
| --- | --- |
| `--run-id` | Run to report on (default: latest) |
| `--list` | List available runs |
| `--json` | Emit metrics as JSON |
| `--out` | Markdown output path |
| `--no-file` | Don't write the Markdown file |

`report` exits non-zero (`3`) if it detects a containment failure, so it can gate
CI.

---

## Where results go

```
results/
├─ results.sqlite3                 # queryable store (one row per attack)
└─ <run_id>/
   ├─ events.jsonl                 # every event, timestamped (nothing is a black box)
   ├─ results.jsonl                # one line per attack: attack + run + verdict
   └─ report.md                    # generated Markdown report
sandbox/
└─ runs/<run_id>/<attack_id>/      # each attack's isolated workspace
```

All of `results/`, `logs/`, and `sandbox/runs/` are git-ignored.

---

## Reproducibility

Each run records the target/attacker/judge model ids, the guard, the egress
allowlist, and a **SHA-256 fingerprint of the exact system prompt** it ran
against — so a whole campaign is provably one prompt version, and changing the
prompt shows up in the results instead of silently shifting the numbers.

Attacker variants are non-deterministic (they come from a model), but every
generated variant records its `parent_id`, so any result traces back to the
hand-written seed it derived from.

---

## Extending

### Add an attack

Append an entry to [`attacker/seeds.yaml`](attacker/seeds.yaml). Give it a
category, a `delivery` (`direct` or `file`), and — for code-judged categories —
a `success_signal` describing the machine-checkable artefact of success.
`tests/test_corpus.py` validates the corpus on every test run.

### Add a defence (Milestone 5)

Implement the `Guard` protocol in [`target_agent/guard.py`](target_agent/guard.py)
and register it in `GUARDS`. The agent loop and tool layer already call every
hook (`on_user_message`, `on_tool_call`, `on_tool_result`, `on_model_output`),
so a guard slots in without touching the loop. Then compare:

```bash
python -m runner --category all --n 10 --guard none   --run-id baseline
python -m runner --category all --n 10 --guard myguard --run-id defended
python -m report --run-id baseline
python -m report --run-id defended
```

---

## Testing

```bash
pytest                       # full suite (no API key or Docker needed)
pytest -m "not docker"       # skip anything requiring a daemon
ruff check .                 # lint
```

The suite runs entirely offline: the agent loop is exercised against a scripted
fake client, and all judging paths are tested with synthetic runs.

## Requirements

- Python 3.11+
- Docker + Docker Compose (for real, sandboxed runs; optional for development)
- An `ANTHROPIC_API_KEY`

## License

MIT.
