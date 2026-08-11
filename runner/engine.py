"""Campaign orchestration.

For each attack the engine:

1. provisions a clean per-attack sandbox directory under ``sandbox/runs/...``
   and lays down any ``setup_files`` / ``setup_symlinks`` the attack needs;
2. dispatches the attack — inside the container when Docker is available, in
   process otherwise;
3. hands the resulting run to the judge;
4. persists attack + run + verdict to the store and mirrors it to JSONL.

Everything is logged with timestamps to ``results/<run_id>/events.jsonl`` so a
whole campaign is reconstructable after the fact.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attacker.campaign import build_campaign
from attacker.generator import VariantGenerator
from attacker.models import CATEGORIES, Attack
from common.config import Settings
from common.ids import new_run_id
from common.llm import make_client
from common.logging import EventLogger
from common.timeutil import utcnow_iso
from judge.judge import Judge
from runner.docker_control import DockerController, DockerError
from runner.local_exec import LocalExecutor
from runner.store import ResultStore
from target_agent.prompts import build_system_prompt, new_canary, prompt_fingerprint

__all__ = ["CampaignEngine", "CampaignConfig", "CampaignSummary"]


@dataclass
class CampaignConfig:
    categories: list[str]
    n_per_category: int
    guard: str = "none"
    include_seed: bool = True
    generate_variants: bool = True
    use_docker: bool | None = None  # None = auto-detect
    enable_llm_judge: bool = True
    run_id: str | None = None


@dataclass
class CampaignSummary:
    run_id: str
    executor: str
    sandboxed: bool
    total: int = 0
    succeeded: int = 0
    errored: int = 0
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)


class CampaignEngine:
    """Runs a campaign end to end."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, config: CampaignConfig) -> CampaignSummary:
        for category in config.categories:
            if category not in CATEGORIES:
                raise ValueError(f"unknown category {category!r}; known: {CATEGORIES}")

        run_id = config.run_id or new_run_id()
        self.settings.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        logger = EventLogger(
            path=self.settings.events_path(run_id),
            run_id=run_id,
            component="runner",
        )
        logger.emit(
            "campaign_start",
            categories=config.categories,
            n_per_category=config.n_per_category,
            guard=config.guard,
            target_model=self.settings.target_model,
        )

        client = make_client(self.settings)

        # Choose an executor.
        controller, executor, sandboxed, exec_info = self._select_executor(config, logger)

        # Build the judge (shares the host client; the target's client is separate).
        judge = Judge.build(
            client=client,
            model=self.settings.judge_model,
            enable_llm=config.enable_llm_judge,
            logger=logger.child("judge"),
            effort=self.settings.target_effort,
        )

        generator = None
        if config.generate_variants:
            generator = VariantGenerator(
                client, model=self.settings.attacker_model, logger=logger.child("attacker")
            )

        store = ResultStore(self.settings.db_path)
        summary = CampaignSummary(
            run_id=run_id,
            executor="docker" if sandboxed else "in-process",
            sandboxed=sandboxed,
        )
        canary = new_canary()
        system_prompt = build_system_prompt(
            sandbox_root=self.settings.container_sandbox_root
            if sandboxed
            else str(self.settings.host_sandbox_dir),
            canary=canary,
        )
        store.record_run(
            run_id,
            {
                "started_at": utcnow_iso(),
                "target_model": self.settings.target_model,
                "attacker_model": self.settings.attacker_model,
                "judge_model": self.settings.judge_model,
                "guard": config.guard,
                "categories": config.categories,
                "n_per_category": config.n_per_category,
                "prompt_sha256": prompt_fingerprint(system_prompt),
                "settings": {
                    "executor": summary.executor,
                    "sandboxed": sandboxed,
                    "llm_backend": self.settings.llm_backend,
                    "target_effort": self.settings.target_effort,
                    "target_max_turns": self.settings.target_max_turns,
                    "egress_allowlist": list(self.settings.egress_allowlist),
                    "llm_judge": config.enable_llm_judge,
                    "generate_variants": config.generate_variants,
                    # Where the *target* reached its model from. For a
                    # containerised Ollama run this is host.docker.internal, not
                    # the host-side localhost recorded above.
                    **exec_info,
                },
            },
        )

        jsonl_path = self.settings.run_dir(run_id) / "results.jsonl"

        try:
            for category in config.categories:
                cat_summary = {"total": 0, "succeeded": 0, "blocked_by_code": 0, "errored": 0}
                attacks = build_campaign(
                    category,
                    config.n_per_category,
                    generator=generator,
                    include_seed=config.include_seed,
                    logger=logger.child("attacker"),
                )
                for attack in attacks:
                    verdict, run = self._execute_and_judge(
                        attack,
                        run_id=run_id,
                        canary=canary,
                        controller=controller,
                        executor=executor,
                        sandboxed=sandboxed,
                        judge=judge,
                        config=config,
                        logger=logger,
                    )
                    store.record_attack(
                        run_id=run_id,
                        attack=attack.to_dict(),
                        run=run,
                        verdict=verdict.to_dict(),
                        jsonl_path=jsonl_path,
                    )
                    cat_summary["total"] += 1
                    summary.total += 1
                    if verdict.success:
                        cat_summary["succeeded"] += 1
                        summary.succeeded += 1
                    if verdict.blocked_by_code:
                        cat_summary["blocked_by_code"] += 1
                    if run.get("error"):
                        cat_summary["errored"] += 1
                        summary.errored += 1
                summary.per_category[category] = cat_summary
                logger.emit("category_done", category=category, **cat_summary)
        finally:
            store.finish_run(run_id)
            store.close()
            if controller is not None:
                controller.down()
            logger.emit("campaign_end", **_summary_fields(summary))
            logger.close()

        return summary

    # -- executor selection -------------------------------------------------

    def _select_executor(
        self, config: CampaignConfig, logger: EventLogger
    ) -> tuple[DockerController | None, LocalExecutor | None, bool, dict[str, Any]]:
        """Pick an executor, and return what the run record should say about it.

        The fourth element describes where the *target* reached its model from —
        for a containerised Ollama run that is ``host.docker.internal``, not the
        host-side ``localhost`` in ``settings``.
        """
        want_docker = config.use_docker
        backend = self.settings.llm_backend
        local_info: dict[str, Any] = (
            {"model_endpoint": self.settings.ollama_base_url} if backend == "ollama" else {}
        )

        controller = DockerController(self.settings, logger=logger.child("docker"))

        if want_docker is False:
            logger.emit(
                "executor_selected",
                executor="in-process",
                reason="use_docker=False",
                llm_backend=backend,
            )
            return None, LocalExecutor(self.settings, logger=logger), False, local_info

        if controller.available():
            try:
                controller.build()
                controller.up()
                selftest = controller.selftest()
                if selftest.returncode != 0:
                    raise DockerError(
                        "in-container containment self-test FAILED; refusing to run. "
                        f"stderr: {selftest.stderr[-800:]}"
                    )
                # The Ollama backend reaches the model over the network, so the
                # container must actually be able to get to the host's daemon
                # (host.docker.internal, mapped in docker-compose.yml). Prove it
                # here rather than discovering it once per attack.
                preflight = self._ollama_preflight(backend, controller, logger)
                docker_info: dict[str, Any] = {}
                if preflight is not None:
                    docker_info = {
                        "model_endpoint": preflight.get("base_url"),
                        "model_endpoint_ip": preflight.get("resolved_ip"),
                    }
                logger.emit(
                    "executor_selected",
                    executor="docker",
                    reason="docker available and self-test passed",
                    selftest="passed",
                    llm_backend=backend,
                    **docker_info,
                )
                return controller, None, True, docker_info
            except DockerError as exc:
                controller.down()
                if want_docker is True:
                    raise
                logger.emit(
                    "executor_fallback",
                    reason=f"docker setup failed: {exc}",
                    executor="in-process",
                )

        if want_docker is True:
            raise DockerError("Docker was required (--docker) but is not available.")

        logger.emit(
            "executor_selected",
            executor="in-process",
            reason="docker unavailable; using in-process fallback (NOT sandboxed)",
            llm_backend=backend,
        )
        return None, LocalExecutor(self.settings, logger=logger), False, local_info

    def _ollama_preflight(
        self, backend: str, controller: DockerController, logger: EventLogger
    ) -> dict[str, Any] | None:
        """Confirm the container can reach the host's Ollama, or raise DockerError.

        Only meaningful for the local backend; the Anthropic path talks to a
        remote endpoint the bridge network already routes to.
        """
        if backend != "ollama":
            return None

        report = controller.ollama_preflight()
        logger.emit("ollama_preflight", **report)
        if report.get("ok"):
            return report

        endpoint = report.get("base_url") or "the configured OLLAMA_BASE_URL"
        stage = report.get("stage", "unknown")
        detail = report.get("error", "")
        if stage == "dns":
            hint = (
                "host.docker.internal did not resolve inside the container. Check the "
                "`extra_hosts: host.docker.internal:host-gateway` entry in "
                "docker/docker-compose.yml, and that the container was recreated after "
                "it was added (`docker compose ... up -d --force-recreate`)."
            )
        else:
            hint = (
                "The name resolved but the connection failed — almost always because "
                "Ollama is listening on loopback only (127.0.0.1:11434), which the "
                "container cannot reach. Restart Ollama bound to the host's other "
                "interfaces (set OLLAMA_HOST=0.0.0.0) and confirm with "
                "`netstat -ano | findstr 11434`."
            )
        raise DockerError(
            f"the container could not reach the Ollama endpoint {endpoint} "
            f"(stage={stage}): {detail}. {hint} "
            f"Alternatively run with --no-docker, which reaches localhost:11434 "
            f"directly but is NOT sandboxed."
        )

    # -- one attack ---------------------------------------------------------

    def _execute_and_judge(
        self,
        attack: Attack,
        *,
        run_id: str,
        canary: str,
        controller: DockerController | None,
        executor: LocalExecutor | None,
        sandboxed: bool,
        judge: Judge,
        config: CampaignConfig,
        logger: EventLogger,
    ) -> tuple[Any, dict[str, Any]]:
        host_sandbox = self.settings.host_attack_dir(run_id, attack.id)
        skipped_links = self._provision_sandbox(host_sandbox, attack, logger)

        if skipped_links:
            # The attack's vector was never staged, so there is nothing to test.
            # Don't spend minutes running it: mark it errored here and let the
            # judge's error short-circuit keep it out of the success-rate
            # denominator. See _provision_sandbox for why.
            logger.emit(
                "attack_not_dispatched",
                attack_id=attack.id,
                category=attack.category,
                reason="symlink_setup_failed",
                links=skipped_links,
            )
            run: dict[str, Any] = {
                "run_id": run_id,
                "attack_id": attack.id,
                "category": attack.category,
                "error": f"symlink_setup_failed: {skipped_links}",
                "tool_calls": [],
                "transcript": [],
                "final_text": "",
            }
        else:
            request = {
                "run_id": run_id,
                "attack_id": attack.id,
                "category": attack.category,
                "prompt": attack.prompt,
                "canary": canary,
                "guard": config.guard,
                "model": self.settings.target_model,
                "max_turns": self.settings.target_max_turns,
                "max_tokens": self.settings.target_max_tokens,
                "effort": self.settings.target_effort,
            }
            logger.emit("attack_dispatch", attack_id=attack.id, category=attack.category)

            if sandboxed and controller is not None:
                container_root = self.settings.container_attack_root(run_id, attack.id)
                # Tell the container which subdir of the bind mount is this attack's root.
                request["sandbox_root"] = container_root
                run = self._run_in_container(controller, request, attack, logger)
            else:
                run = executor.run_attack(request, sandbox_dir=host_sandbox)  # type: ignore[union-attr]

        run.setdefault("run_id", run_id)
        run["sandboxed"] = sandboxed
        run["canary"] = canary

        verdict = judge.judge_run(attack, run, sandbox_dir=str(host_sandbox))
        return verdict, run

    def _run_in_container(
        self,
        controller: DockerController,
        request: dict[str, Any],
        attack: Attack,
        logger: EventLogger,
    ) -> dict[str, Any]:
        # The image sets AGENT_SANDBOX_ROOT=/sandbox; the per-attack subdir of
        # the bind mount is passed as --sandbox-root so each attack gets a clean,
        # isolated root inside the container.
        try:
            return controller.run_attack(
                request, sandbox_root=request.get("sandbox_root")
            )
        except DockerError as exc:
            logger.emit("attack_container_error", attack_id=attack.id, error=str(exc))
            return {
                "run_id": request["run_id"],
                "attack_id": attack.id,
                "category": attack.category,
                "error": f"container_error: {exc}",
                "tool_calls": [],
                "transcript": [],
                "final_text": "",
            }

    # -- sandbox provisioning ----------------------------------------------

    @staticmethod
    def _provision_sandbox(host_sandbox: Path, attack: Attack, logger: EventLogger) -> list[str]:
        """Lay down the attack's workspace; return the links that could not be made.

        A non-empty return value means the sandbox does not carry the state the
        attack needs, and the caller must not dispatch it.
        """
        if host_sandbox.exists():
            shutil.rmtree(host_sandbox, ignore_errors=True)
        host_sandbox.mkdir(parents=True, exist_ok=True)

        for rel_path, content in attack.setup_files.items():
            target = host_sandbox / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        skipped_links: list[str] = []
        for link_name, link_target in attack.setup_symlinks.items():
            link_path = host_sandbox / link_name
            try:
                link_path.parent.mkdir(parents=True, exist_ok=True)
                link_path.symlink_to(link_target)
            except (OSError, NotImplementedError) as exc:
                # Symlink creation can fail on Windows without the privilege
                # (WinError 1314). When it does, the symlink-escape vector this
                # attack is built around simply does not exist in the sandbox:
                # there is no link to follow out of the root. Running the attack
                # anyway would let the model "resist" a vector that was never
                # present, and the judge would score that as a defended attack —
                # inflating the defence rate with a test that never happened.
                # So report the failure up and let the caller skip the dispatch.
                logger.emit(
                    "symlink_setup_skipped",
                    attack_id=attack.id,
                    link=link_name,
                    target=link_target,
                    error=str(exc),
                )
                skipped_links.append(link_name)
        return skipped_links


def _summary_fields(summary: CampaignSummary) -> dict[str, Any]:
    return {
        "total": summary.total,
        "succeeded": summary.succeeded,
        "errored": summary.errored,
        "executor": summary.executor,
        "sandboxed": summary.sandboxed,
    }
