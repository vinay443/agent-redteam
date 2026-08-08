# SAFETY.md — containment model and how to verify it

`agent-redteam` deliberately provokes an autonomous, tool-using LLM into
misbehaving. That is only responsible if the agent's misbehaviour is boxed. This
document states the guarantees, where each one is enforced, and the exact
command that proves it.

The governing principle: **every boundary is enforced in code, not in the system
prompt.** The prompt is the thing under test — the lab measures how often it can
be talked around. It is never the thing keeping you safe.

## Threat model

The "attacker" is not a person on your network. It is a corpus of prompts and
files (`attacker/seeds.yaml` plus LLM-generated variants) fed to the target
agent, whose goal is to make the agent:

- write or read files outside its workspace (**permission escalation**),
- follow instructions embedded in file content (**prompt injection**),
- abandon the user's task for the attacker's (**goal hijacking**), or
- reveal its own system prompt (**system-prompt exfiltration**).

The target agent has real tool permissions (`read_file`, `write_file`). We
assume it *will* try to do what the attack asks — the interesting question is
what stops it when it does.

## Defence in depth: three layers

| Layer | Enforced in | Defeats | Load-bearing? |
| --- | --- | --- | --- |
| 1. Container isolation | `docker/docker-compose.yml` | A tool escape, a runaway loop, a fork bomb reaching the host | **Yes** |
| 2. Tool boundary | `target_agent/sandbox.py`, `target_agent/net.py` | Any path or URL leaving the sandbox / allowlist | **Yes** |
| 3. System prompt | `target_agent/prompts.py` | Nothing on its own — it states intent and is what we measure | **No** |

If layer 3 fails, that is a *finding*. If layer 1 or 2 fails, that is a *bug*,
and the report flags it in red.

---

## Layer 1 — container isolation

Defined entirely in [`docker/docker-compose.yml`](docker/docker-compose.yml).
Each requirement, and how to confirm it:

### No host mounts except `./sandbox`

The only `volumes:` entry is `../sandbox:/sandbox:rw`. There is no source-code
mount, no config mount, and — critically — **no Docker socket** (`/var/run/docker.sock`
is never mounted; mounting it would hand the container control of the daemon).

```bash
# Should list exactly one bind mount, Source ending in /sandbox:
docker inspect agent-redteam-target \
  --format '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}} ({{.Mode}}){{println}}{{end}}'
```

### No privileged mode, all capabilities dropped

```yaml
security_opt: [ "no-new-privileges:true" ]
cap_drop: [ ALL ]
```

`privileged`, `cap_add`, and `devices` never appear.

```bash
docker inspect agent-redteam-target \
  --format 'Privileged={{.HostConfig.Privileged}} CapAdd={{.HostConfig.CapAdd}} CapDrop={{.HostConfig.CapDrop}}'
# Expect: Privileged=false CapAdd=[] CapDrop=[ALL]
```

### Resource limits (a runaway loop / fork bomb cannot affect the host)

```yaml
mem_limit: 512m
memswap_limit: 512m   # no swap beyond the cap
cpus: 1.0
pids_limit: 128       # process-count cap — fork bombs die here
```

```bash
docker inspect agent-redteam-target \
  --format 'Mem={{.HostConfig.Memory}} Swap={{.HostConfig.MemorySwap}} PidsLimit={{.HostConfig.PidsLimit}} NanoCpus={{.HostConfig.NanoCpus}}'
# Expect: Mem=536870912 Swap=536870912 PidsLimit=128 NanoCpus=1000000000
```

### Read-only root filesystem

The container's root FS is read-only; the agent can write only to the
bind-mounted `/sandbox` and two small, ephemeral tmpfs scratch areas.

```bash
docker inspect agent-redteam-target --format 'ReadonlyRootfs={{.HostConfig.ReadonlyRootfs}}'
# Expect: ReadonlyRootfs=true
```

### Non-root user, no inbound ports

The image runs as UID 10001 (`docker/Dockerfile`), and the compose file exposes
no `ports:` — nothing listens for inbound connections. The runner drives the
agent over `docker exec -i` (stdin/stdout), so there is no network service to
attack.

### Host network beyond an explicit allowlist

The container is on a `bridge` network so it can reach the Anthropic API. Docker
Compose cannot express a *per-host* allowlist, so the allowlist is enforced
**in-process** (layer 2, below). For a stricter posture, front the container
with a filtering egress proxy and set `network_mode: none` in the compose file.

---

## Layer 2 — the tool boundary

### Filesystem containment (`target_agent/sandbox.py`)

`resolve_in_sandbox()` decides, for every path the agent supplies, whether it
stays inside the workspace. The rules:

1. **Compare on `os.path.realpath`, not string prefixes.** `realpath` resolves
   `..` segments *and* follows symlinks, then containment is decided by path-
   component comparison (`Path.parents`). String matching is defeated by `../`,
   by a planted symlink, and by a sibling directory whose name merely starts
   with the root's (`/sandbox-evil` vs `/sandbox`) — all of which this blocks.
2. **Reject, never clamp.** An out-of-bounds path raises `SandboxViolation`;
   it is never silently rewritten to something safe, because that would hide the
   attack from the metrics.
3. **Re-verify after writing** (`assert_within`) to close the resolve→open race.

### Network egress (`target_agent/net.py`)

There is **no network tool** for the target agent in this pass. The allowlist
still exists and is enforced today on the agent's own API `base_url` at client
construction (`common/llm.make_client`), so a tampered `ANTHROPIC_BASE_URL`
cannot redirect traffic. It is an allowlist, not a filter: HTTPS only, no
credentials in the URL, exact host or subdomain match, no bare IPs, default port
only. Any future `web_fetch` tool **must** reuse `assert_url_allowed` rather than
grow its own check.

---

## Verifying the tool boundary — the path-traversal test

This is the headline safety check the task requires. It asserts that a path-
traversal attempt is **blocked** (raises), not merely warned about.

```bash
# The whole safety-critical surface:
pytest tests/test_sandbox.py tests/test_tools.py tests/test_net.py -v
```

`tests/test_sandbox.py` proves, among others, that these all raise
`SandboxViolation`:

- `../escape.txt` and `../../../../etc/passwd` (relative traversal),
- `/etc/passwd` (absolute path outside the root),
- `notes/../../escape.txt` (traversal hidden mid-path),
- a symlink inside the sandbox pointing at `/etc` (symlink escape),
- `/sandbox-evil/...` (prefix-confusion sibling),

while legitimate nested paths inside the root resolve normally.

`tests/test_tools.py` proves the tool layer surfaces an escape attempt as
`ok=False, blocked=True, outside_root=True` — a recorded, non-silent refusal —
and that nothing is written outside the root.

### The same check, inside the container

Run the containment self-test in the *actual* runtime environment (this is what
the runner executes automatically before every campaign and refuses to proceed
if it fails):

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker exec -i agent-redteam-target python -m target_agent --selftest
docker compose -f docker/docker-compose.yml down -v
# Expect: "selftest: N/N passed" and exit code 0
```

The in-container self-test additionally exercises the **symlink** escape vector,
which host-side unit tests skip on Windows without symlink privilege.

---

## Runner guarantees

- Before any campaign, the runner builds the image, starts the container, and
  runs the in-container self-test. **If containment fails, the campaign aborts.**
  (`runner/engine.py :: _select_executor`.)
- Each attack runs against its own freshly-provisioned subdirectory of the bind
  mount, so attacks cannot see each other's artefacts.
- If Docker is unavailable, the runner can fall back to in-process execution —
  but every such result is tagged `sandboxed=False` and the report prints a
  warning. **In-process runs have no runtime isolation and are for development
  only.** Use `--docker` to require containment and fail otherwise.

---

## What is intentionally NOT here

Per the project scope, and not to be added without an explicit decision:

- **No shell or code-execution tool** for the target agent.
- **No network access** for the target agent beyond the API base URL.
- **Nothing that touches the host filesystem outside `./sandbox`.**

The defence hook (`target_agent/guard.py`) is the seam where a guard layer will
be added later; it changes none of the guarantees above.

## If you find a containment failure

A non-zero `escaping_calls` count in the report, or a failing `--selftest`,
means layer 1 or 2 has a hole. Treat it as a security bug: stop running
campaigns, fix `target_agent/sandbox.py` (or the compose file), add a regression
test to `tests/test_sandbox.py`, and re-run the self-test before continuing.
