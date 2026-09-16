"""The loop: poll, gate, spawn, publish, notify.

Everything stateful about a review lives here; the modules it calls are either
pure (gates, anchor, contract) or a single narrow surface (publish, slack,
runner). That split is what makes the policy testable without a GitHub account.

The other two phases of a tick are their own modules: `threads` answers replies to
earlier findings and `ci_watch` reads the build an approval paid for. Both borrow
`_slot` from here — capacity for a container, already gated — since they spend the
same money out of the same cap.

Concurrency is per PR, capped by `max_concurrent_reviews`. A slow review cannot
delay the others or collide with the next tick, because a tick only schedules work
the semaphore has room for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, NamedTuple

from robbie import budget, props_bridge, publish
from robbie import slack as slackmod
from robbie.anchor import parse_findings, severity_count, summary_findings
from robbie.ci_watch import CiWatch
from robbie.config import Choice, Config, RepoConfig, Secrets
from robbie.contract import Blocks, preamble, threads_block
from robbie.db import Db, now_ms
from robbie.gates import Decision, already_judged, dedup_key, done_label, evaluate, label_hold
from robbie.github import (
    QUEUE_LIMIT,
    GhError,
    PrMeta,
    Thread,
    authored,
    ci_started,
    labeled_heads,
    last_review_request,
    my_threads,
    pr_meta,
    queue,
    stale_changes_requested,
    standing_rejection,
    summarize_checks,
    whoami,
)
from robbie.outcome import Outcome, Slot
from robbie.runner import ReviewRun, prune_transcripts, run_review
from robbie.slack import Slack
from robbie.threads import Sweeper

logger = logging.getLogger(__name__)


async def candidates(repo: RepoConfig) -> list[int]:
    """The daemon's per-repo candidate set: pending review requests, plus (when
    self_review) the reviewer's own labeled PRs — order-preserving union."""
    prs = await queue(repo.slug, label=repo.label, reviewer=repo.reviewer_login)
    if repo.self_review:
        own = await authored(repo.slug, label=repo.label, author=repo.reviewer_login)
        prs += [p for p in own if p not in prs]
    return prs


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        secrets: Secrets,
        db: Db,
        slack: Slack,
        *,
        dry_run: bool = False,
        no_publish: bool = False,
        model: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.db = db
        self.slack = slack
        self.dry_run = dry_run
        # unlike dry_run, the review still runs; only the outward writes stop
        self.no_publish = no_publish
        self.model = model  # `once --model` only; see Secrets.review_base_url
        self._self_login: str | None = None
        # (repo, pr) → what the gate read, handed to the prompt in the same tick
        self._threads: dict[tuple[str, int], list[Thread]] = {}
        self._inflight = 0  # containers spending right now, which no gate can see
        self._sem = asyncio.Semaphore(cfg.max_concurrent_reviews)
        self._gate_sem = asyncio.Semaphore(cfg.max_concurrent_checks)
        # Deciding and reserving under one lock: anything that suspends between
        # them lets every waiting review read the same pre-reserve number and
        # admit itself, the stampede the reserves exist to prevent.
        self._admit_lock = asyncio.Lock()

    # Built per call rather than held: both are frozen views over this object's
    # own state, and `dry_run` / `no_publish` can be flipped after construction.

    @property
    def sweeper(self) -> Sweeper:
        return Sweeper(
            self.cfg, self.secrets, self.db, self.slack, self._slot, self._gate_sem,
            dry_run=self.dry_run, no_publish=self.no_publish,
        )

    @property
    def ci(self) -> CiWatch:
        return CiWatch(
            self.cfg, self.db, self._gate_sem,
            dry_run=self.dry_run, no_publish=self.no_publish,
        )

    @property
    def _quiet(self) -> bool:
        """This pass writes nothing outward, so it may record nothing either."""
        return self.dry_run or self.no_publish

    def _pass_state(self, state: str, reason: str | None = None) -> dict[str, Any]:
        """How a finished pass is recorded, given that `--no-publish` posts nothing.

        The row keeps its cost, model and counts, but not the state: every gate
        reads `published` and `held` as judged, and a review nobody saw must not
        park the PR out of the queue.
        """
        if self.no_publish:
            return {"state": "failed", "hold_reason": "--no-publish: nothing was posted"}
        return {"state": state, "hold_reason": reason}

    def _creds_expired(self) -> tuple[int, int]:
        """(ms past expiry, expiresAt) for the mounted account token — (0, 0) when
        fresh or not applicable. While past expiry, the proxy refuses every model
        call, so a container spawned now is a guaranteed failure and a guaranteed
        operator DM per tick (2026-09-15: seven DMs in an hour for one stale
        token). An unreadable file is the startup check's complaint, not ours.
        """
        if self.cfg.backend != "oauth" or self.secrets.claude_credentials is None:
            return 0, 0
        try:
            creds = json.loads(self.secrets.claude_credentials.read_text())
            exp = int(float(creds["claudeAiOauth"].get("expiresAt", 0)))
        except Exception:  # noqa: BLE001
            return 0, 0
        over = now_ms() - exp
        return (over, exp) if over > 0 else (0, 0)

    # ----- entry points --------------------------------------------------

    async def poll_once(self) -> list[Outcome]:
        """One tick across every configured repo: answer replies, then review.

        In that order because one `my_threads` read per PR feeds both gate 5 and
        the prompt's prior-conversation block. Reversed, the gate rules on threads
        this tick is about to close and a re-review re-raises conceded findings.
        """
        self._threads.clear()
        if not self.dry_run:  # housekeeping, so it belongs to a real tick only
            prune_transcripts(self.cfg)
        stale_ms, token_exp = self._creds_expired()
        if stale_ms:
            # one DM per token, not per tick: a refresh mints a new expiresAt,
            # so the next outage announces itself exactly once too
            await self._dm_owner_once(
                f"creds-expired:{token_exp}",
                f"The account's access token expired {stale_ms // 60_000} min ago, so "
                "reviews are paused — every container would fail until it refreshes. "
                "Run any Claude session on the host (creds-sync carries it over within "
                "minutes); I'll start again on my own.",
            )
            logger.warning(
                "credentials expired %d min ago — skipping containers this tick",
                stale_ms // 60_000,
            )
        answered: list[Outcome] = []
        if not stale_ms and self._meter().allowed:
            answered = await self.answer_threads()  # a container, so the same spend gate
        answered += await self.ci.watch()  # gh reads only, so no gate of its own
        jobs: list[asyncio.Task[Outcome]] = []
        async with asyncio.TaskGroup() as tg:
            for repo in self.cfg.repos:
                try:
                    prs = await candidates(repo)
                except GhError as ex:
                    logger.warning("could not fetch the queue for %s: %s", repo.slug, ex)
                    continue
                logger.info("%s: %d PR(s) in queue", repo.slug, len(prs))
                # the panel's other half: what a human is actually still waiting on
                if not self.dry_run:
                    self.db.set_requested(repo.slug, prs)

                if not self.db.is_seeded(repo.slug):
                    await self._seed(repo, prs)
                    continue

                await self._retire_unlabeled(repo)
                if stale_ms:
                    continue  # board bookkeeping done; no containers until creds refresh
                for pr in prs:
                    jobs.append(tg.create_task(self._handle(repo, pr)))
        outcomes = answered + [job.result() for job in jobs]
        await self._heartbeat()
        return outcomes

    async def _heartbeat(self) -> None:
        """The board's one-glance reviewer card: what the panel would say, pushed
        once per tick over the bridge. Best-effort like every bridge call; quiet
        runs say nothing anywhere, the board included."""
        if self._quiet or not self.cfg.props_url:
            return
        midnight = budget.midnight_ms()
        gate = self._meter()
        stale_ms, _ = self._creds_expired()
        await props_bridge.heartbeat(self.cfg.props_url, {
            "at": now_ms(),
            "paused": (f"credentials expired {stale_ms // 60_000}m ago"
                       if stale_ms else None),
            "tick_s": self.cfg.poll_interval_s,
            "reviewing": [int(r["pr"]) for r in self.db.unfinished(10)
                          if r["state"] == "running"],
            "today": self.db.verdicts_since(midnight),
            "failed_today": self.db.failed_runs_since(midnight),
            "spend_usd": round(self.db.spend_since(midnight, self.cfg.endpoint_models), 2),
            "budget": {"allowed": gate.allowed, "detail": gate.detail},
            "ready": [int(r["pr"]) for r in self.db.approved_and_green(
                midnight - 30 * 86_400_000) if r["ci_state"] == "green"],
        })

    async def review_one(self, slug: str, pr: int) -> Outcome:
        """Force a review, ignoring the queue, the gates and prior state."""
        repo = self.cfg.repo(slug)
        meta = await pr_meta(slug, pr)
        requested_at = await _requested_at_or_blank(slug, pr, repo.reviewer_login)
        key = dedup_key(slug, pr, meta.head_sha, requested_at)
        if self.model:
            # one row per model on the same commit; the key is what rows replace on
            key = f"{key}:{self.model}"
        return await self._review(repo, meta, key, requested_at)

    async def status(self) -> list[str]:
        rows: list[str] = []
        for repo in self.cfg.repos:
            prs = await candidates(repo)
            rows += await asyncio.gather(*(self._status_row(repo, pr) for pr in prs))
        return rows

    async def _status_row(self, repo: RepoConfig, pr: int) -> str:
        async with self._gate_sem:  # two gh calls each; a full queue is a lot at once
            meta = await pr_meta(repo.slug, pr)
            # through the gates, not a second reading of the same labels
            if (held := label_hold(meta, repo)) is not None:
                mark = f"⛔ {held.reason}"
            elif (done := done_label(meta, repo)) is not None:
                mark = f"✓ {done} — a human has it"
            else:
                requested_at = await _requested_at_or_blank(repo.slug, pr, repo.reviewer_login)
                prior = self.db.get_review(
                    dedup_key(repo.slug, pr, meta.head_sha, requested_at)
                )
                if prior is None and self.db.sha_was_judged(repo.slug, pr, meta.head_sha):
                    mark = "↻ needs re-review (re-requested since last pass)"
                elif prior is None:
                    mark = "• not reviewed yet"
                elif prior.state == "held":
                    mark = f"⏸ held — {prior.hold_reason}"
                elif prior.state == "published":
                    mark = f"✓ reviewed ({prior.verdict})"
                else:
                    mark = f"… {prior.state}"
        return f"  {repo.slug}#{pr:<6} {mark}  — {meta.title[:60]}"

    async def answer_threads(
        self, slug: str | None = None, only: tuple[int, ...] = ()
    ) -> list[Outcome]:
        """The reply sweep. Lives in `threads`; `robbie threads` starts here."""
        return await self.sweeper.sweep(slug, only)

    # ----- per-PR pipeline ----------------------------------------------

    async def _handle(self, repo: RepoConfig, pr: int) -> Outcome:
        try:
            return await self._handle_inner(repo, pr)
        except GhError as ex:
            # one unreachable PR must not take down the tick
            logger.warning("%s#%s: github unreachable: %s", repo.slug, pr, ex)
            return Outcome(repo.slug, pr, "skip", "github unreachable")
        except Exception as ex:  # noqa: BLE001 — same reasoning, wider net
            logger.exception("%s#%s: unhandled error", repo.slug, pr)
            return Outcome(repo.slug, pr, "failed", str(ex))

    async def _handle_inner(self, repo: RepoConfig, pr: int) -> Outcome:
        # the gate slot is released before the review starts. Holding it across a
        # 30-minute container would turn this cap into the total-in-flight cap,
        # and PRs past it would go unchecked until a review finished.
        async with self._gate_sem:
            meta, key, requested_at, decision = await self._gate(repo, pr)
        return await self._act(repo, meta, key, requested_at, decision)

    async def _gate(
        self, repo: RepoConfig, pr: int
    ) -> tuple[PrMeta, str, str, Decision]:
        meta = await pr_meta(repo.slug, pr)

        # each short-circuit skips the API call the next line would have made
        if (held := label_hold(meta, repo)) is not None:
            return meta, "", "", held

        # before the expensive one: `last_review_request` pages the whole issue
        # timeline, and an approved PR sits in the queue until somebody merges it
        if (done := done_label(meta, repo)) is not None:
            if not self.dry_run and self.db.settle_done(repo.slug, pr):
                logger.info("%s#%s: %s — a human has it; done reviewing", repo.slug, pr, done)
            return meta, "", "", Decision("skip", f"{done} — a human has it")

        requested_at = await last_review_request(repo.slug, pr, repo.reviewer_login)
        key = dedup_key(repo.slug, pr, meta.head_sha, requested_at)
        prior = self.db.get_review(key)
        if (judged := already_judged(prior.state if prior else None)) is not None:
            return meta, key, requested_at, judged

        # one query serves both the gate and the prompt's prior-conversation block
        threads = (await my_threads(repo.slug, pr, repo.reviewer_login)).threads
        self._threads[repo.slug, pr] = threads
        return meta, key, requested_at, evaluate(
            meta,
            repo,
            sha_judged=self.db.sha_was_judged(repo.slug, pr, meta.head_sha),
            open_threads=sum(1 for t in threads if t.awaiting_author),
        )

    async def _act(
        self,
        repo: RepoConfig,
        meta: PrMeta,
        key: str,
        requested_at: str,
        decision: Decision,
    ) -> Outcome:
        if decision.action == "skip":
            return Outcome(repo.slug, meta.number, "skip", decision.reason)

        if decision.action == "hold":
            if decision.dm:
                await self._dm_owner_once(f"hold:{key}", decision.dm)
            # a later gate reads this row as judged, so a pass that told nobody
            # must not write it
            if decision.record and not self._quiet:
                self.db.record_hold(
                    key=key, repo=repo.slug, pr=meta.number, head_sha=meta.head_sha,
                    requested_at=requested_at, reason=decision.reason,
                )
            return Outcome(repo.slug, meta.number, "hold", decision.reason)

        if decision.action == "ci-note":
            result = await publish.once_per(
                self.db,
                publish.posted_key("ci-red", repo, meta),
                lambda: publish.post_ci_note(
                    repo, meta, decision.checks, dry_run=self._quiet
                ),
                quiet=self._quiet,
            )
            if result is None:
                return Outcome(
                    repo.slug, meta.number, "ci-note",
                    f"already noted the red build for {meta.head_sha[:8]}",
                )
            return Outcome(repo.slug, meta.number, "ci-note", result.detail)

        # the spend gate is `_slot`'s alone; asking here too is two places
        # deciding one thing
        return await self._review(repo, meta, key, requested_at)

    def _choice(self, key: str) -> Choice:
        """Which model reviews this key, and therefore whose meter it spends."""
        if self.model:
            return self.cfg.named_model(self.model)
        return self.cfg.choose_model(key)

    @contextlib.asynccontextmanager
    async def _slot(self, key: str | None = None) -> AsyncIterator[Slot]:
        """Capacity to run one container: a semaphore slot and a spend reserve.

        `key` picks the arm and therefore the meter; without one the run is not a
        review and goes on the account's. The one place the spend gate is asked,
        and so the one place that tells the operator about it.
        """
        async with self._sem:
            async with self._admit_lock:
                choice, gate = (
                    await self._admit(key) if key is not None
                    else (Choice(), self._meter())
                )
                if gate.allowed:
                    self._inflight += 1
            await self._tell_owner(gate)
            if not gate.allowed:
                yield Slot(False, gate.detail)
                return
            try:
                yield Slot(True, gate.detail, choice)
            finally:
                self._inflight -= 1

    async def _tell_owner(self, gate: budget.Verdict) -> None:
        """What a spend verdict is worth waking the operator for, at most once each:
        reviews have stopped, or a review ran with nothing measuring it."""
        if gate.notice_key is None:
            return
        if not gate.allowed:
            await self._dm_owner_once(
                gate.notice_key,
                f"Holding off on reviews — {gate.detail}. I'll start again on my own.",
            )
        # either meter, not just the account's: an unguarded run is unguarded
        # whoever was supposed to be measuring it
        elif budget.UNREADABLE in gate.notice_key:
            await self._dm_owner_once(
                gate.notice_key, f"I can't read the spend budget: {gate.detail}"
            )

    def _meter(self, *, via_endpoint: bool = False) -> budget.Verdict:
        """The spend gate. A SQLite read now that the poller owns the HTTP."""
        return budget.check(self.cfg, self.db, self._inflight, via_endpoint=via_endpoint)

    async def _admit(self, key: str) -> tuple[Choice, budget.Verdict]:
        """The arm that will review this key, and whether its meter allows it.

        The arms cover for each other, trading the exact ratio for coverage. A model
        named on the CLI is never substituted — that was a request.
        """
        first = self._choice(key)
        verdict = self._meter(via_endpoint=first.via_endpoint)
        if verdict.allowed or self.model:
            return first, verdict
        other = self.cfg.fallback_for(first)
        if other is None:
            return first, verdict
        spare = self._meter(via_endpoint=other.via_endpoint)
        if not spare.allowed:
            return first, verdict
        logger.info(
            "%s: %s has no room (%s), falling back to %s",
            key, first.model or "the account", verdict.detail, other.model,
        )
        return other, spare

    async def _dm_owner_once(self, key: str, text: str) -> None:
        """One DM per key, ever — and a quiet run must not spend the key, or the
        next real tick has nothing left to say about every currently-held PR."""
        if self.dry_run or self.no_publish:
            if not self.db.notice_seen(key):
                await self.slack.dm_owner(text)  # this Slack only logs
            return
        if self.db.notice_once(key):
            await self.slack.dm_owner(text)

    async def _review(
        self, repo: RepoConfig, meta: PrMeta, key: str, requested_at: str
    ) -> Outcome:
        if self.dry_run:
            logger.info("DRY would review %s#%s (%s)", repo.slug, meta.number, key)
            return Outcome(repo.slug, meta.number, "review", "dry run")

        # built before the semaphore: a container slot must not be held open
        # while an API call for the prior conversation is in flight
        prompt = preamble(
            author=meta.author,
            ci=summarize_checks(meta),
            threads=threads_block(await self._prior_threads(repo, meta)),
            history=self._pass_history(repo, meta),
        )

        # the gate ruled minutes ago, behind however many reviews queued here
        async with self._slot(key) as slot:
            if not slot.ok:
                logger.info("budget closed while %s#%s waited: %s",
                            repo.slug, meta.number, slot.detail)
                return Outcome(repo.slug, meta.number, "budget", slot.detail)
            choice = slot.choice
            self.db.start_review(
                key=key, repo=repo.slug, pr=meta.number,
                head_sha=meta.head_sha, requested_at=requested_at,
            )
            await self._tell_props(meta, "reviewing")
            run = await run_review(
                self.cfg, self.secrets, repo, meta, prompt=prompt,
                model=choice.model, via_endpoint=choice.via_endpoint,
            )

        if not run.ok:
            # Not judged, so the next tick retries — and the retry reuses this key,
            # so `INSERT OR REPLACE` erases this row. Append-only `spend` is the
            # only place a failure outlives the attempt that fixed it.
            self.db.record_spend(
                repo=repo.slug, pr=meta.number, kind="failed",
                # an endpoint arm's cost is the CLI's price table, not the
                # account's bill, and `spend` carries no model to exclude it by
                cost_usd=None if choice.via_endpoint else run.cost_usd,
                duration_s=run.duration_s,
            )
            self.db.finish_review(
                key, state="failed", hold_reason=run.error, duration_s=run.duration_s,
                transcript=str(run.transcript or ""), model=choice.model,
            )
            return await self._gave_up(
                repo, meta, run.error or "unknown",
                f"I tried to review *{meta.title}* ({meta.url}) but the run failed: "
                f"{run.error}. I'll retry next cycle.",
            )

        return await self._deliver(repo, meta, key, run, choice)

    async def _deliver(
        self, repo: RepoConfig, meta: PrMeta, key: str, run: ReviewRun, choice: Choice
    ) -> Outcome:
        """What a finished run comes to: record what it cost, then post what it said.

        The review is paid for by the time it gets here, so every way out leaves a
        transcript and tells the operator where to find it.
        """
        blocks = run.blocks
        if blocks is None:  # mode="review" always parses; a caller could still lie
            return await self._gave_up(
                repo, meta, "no blocks",
                f"I ran a review of *{meta.title}* ({meta.url}) that parsed no blocks "
                f"at all. Transcript: {run.transcript}",
            )
        findings = parse_findings(blocks.inline)
        # heterogeneous on purpose — it is the column set every exit below writes
        common: dict[str, Any] = {
            "cost_usd": run.cost_usd, "tokens_in": run.tokens_in,
            "tokens_out": run.tokens_out, "duration_s": run.duration_s,
            "transcript": str(run.transcript or ""), "model": choice.model,
            "findings": len(findings),
            "blocking": severity_count(findings, blocking=True),
            "should_fix": severity_count(findings, blocking=False),
            "summary_findings": summary_findings(blocks.github),
        }

        if (bad := _unusable(blocks)) is not None:
            self.db.finish_review(
                key, verdict=blocks.verdict, **self._pass_state("held", bad.reason), **common
            )
            return await self._gave_up(
                repo, meta, bad.detail,
                f"I reviewed *{meta.title}* ({meta.url}) but {bad.told}, so I posted "
                f"nothing. Transcript: {run.transcript}",
            )
        # _unusable returns for a missing verdict, so past it this is one of the three
        verdict = blocks.verdict
        assert verdict is not None

        if verdict == "ok":
            await publish.clear_needs_work(repo, meta.number, dry_run=self.no_publish)
            await self._retract_rejection(repo, meta)
            ci = await self._request_ci(repo, meta)
            self.db.finish_review(
                key, verdict="ok", **self._pass_state("published"),
                # nothing asked CI on a run that publishes nothing, so nothing to wait for
                ci_state=None if self.no_publish else "waiting",
                # an ok posts no comment at all, and saying so is what lets the
                # reply sweep skip the PR: left unwritten it reads as unknown
                inline=0,
                **common,
            )
            await self._announce_approval(meta, ci)
            await self._tell_props(meta, "posted")
            return Outcome(repo.slug, meta.number, "review", f"ok — {ci}")

        # recorded as judged either way: retrying a permanent publish failure
        # would burn a full review every tick
        self.db.finish_review(key, verdict=verdict, **self._pass_state("published"), **common)
        # resolved before the lambda, which cannot await: it is one `whoami` per
        # process either way, since `_token_login` caches what it resolved
        self_login = await self._token_login()
        try:
            result = await publish.once_per(
                self.db,
                publish.posted_key("review", repo, meta),
                lambda: publish.publish_review(
                    verdict, repo, meta, body=blocks.github, findings=findings,
                    dry_run=self.no_publish, self_login=self_login,
                ),
                quiet=self.no_publish,
            )
        except Exception as ex:  # noqa: BLE001 — the review is done; only delivery failed
            logger.exception("publish failed for %s#%s", repo.slug, meta.number)
            return await self._gave_up(
                repo, meta, f"publish: {ex}",
                f"I couldn't publish my {verdict} review of *{meta.title}* "
                f"({meta.url}): {ex}. It's ready to post by hand: {run.transcript}",
            )
        if result is None:
            return Outcome(
                repo.slug, meta.number, "review",
                f"already published for {meta.head_sha[:8]}",
            )

        if result.posted:
            # how many of them reached a diff line, which is not the same number
            self.db.finish_review(
                key, verdict=verdict, **self._pass_state("published"),
                inline=result.inline, **common,
            )
            await self._notify(repo, meta, verdict, findings)
            await self._tell_props(meta, "posted")
        return Outcome(repo.slug, meta.number, "review", f"{verdict}: {result.detail}")

    async def _gave_up(
        self, repo: RepoConfig, meta: PrMeta, detail: str, told: str
    ) -> Outcome:
        """A paid-for review that reached nobody. Tell the operator, say so upward.

        Not `_dm_owner_once`: a second failure on the same PR is news again.
        """
        await self.slack.dm_owner(told)
        await self._tell_props(meta, "skipped")
        return Outcome(repo.slug, meta.number, "failed", detail)

    async def _tell_props(self, meta: PrMeta, status: str) -> None:
        """The board hears where this head is in its life. Quiet runs report
        nothing, like everything else they touch."""
        if not self._quiet:
            await props_bridge.report(
                self.cfg.props_url, meta.number, meta.head_sha, status
            )

    async def _prior_threads(self, repo: RepoConfig, meta: PrMeta) -> list[Thread]:
        """Reuse what the gate fetched; fetch it for a forced run that skipped it."""
        cached = self._threads.pop((repo.slug, meta.number), None)
        if cached is not None:
            return cached
        try:
            return (await my_threads(repo.slug, meta.number, repo.reviewer_login)).threads
        except GhError:
            logger.warning("%s#%s: could not read prior threads", repo.slug, meta.number)
            return []

    def _pass_history(self, repo: RepoConfig, meta: PrMeta) -> str:
        rows = self.db.passes_for(repo.slug, meta.number)
        if not rows:
            return ""
        past = ", ".join(
            f"{r['head_sha'][:8]} → {r['verdict'] or r['hold_reason'] or r['state']}"
            for r in rows
        )
        return f"This is pass {len(rows) + 1} on this PR. Earlier passes: {past}.\n"

    async def _dismiss_if_vetoed(self, repo: RepoConfig, pr: int) -> str | None:
        """Retract our own standing changes-requested on this PR. None if there is none."""
        node_id = await standing_rejection(repo.slug, pr, repo.reviewer_login)
        if node_id is None:
            return None
        result = await publish.dismiss_own_rejection(
            node_id, dry_run=self.dry_run or self.no_publish
        )
        return result.detail

    async def retract_stale_rejections(
        self, repo: RepoConfig, *, only: tuple[int, ...] = ()
    ) -> list[Outcome]:
        """Dismiss rejections of ours that a later `ok` already contradicted.

        The ok path retracts as it goes; this is the one-shot for the approvals that
        landed before it did. One search names every PR still carrying a standing
        changes-requested of ours; the database says which of those we went on to
        approve. A PR whose newest pass is still a needs-work keeps its veto — that
        one is true.

        `only` is the escape hatch for a PR whose `ok` is not evidence: an eval PR
        collects verdicts from arms picked to be wrong, so its approval must not be
        allowed to retract a rejection that was right.
        """
        try:
            vetoed = {
                int(node["number"])
                for node in await stale_changes_requested(
                    repo.slug, repo.reviewer_login, label=repo.label
                )
            }
        except GhError as ex:
            logger.warning("could not read %s's standing rejections: %s", repo.slug, ex)
            return []
        contradicted = sorted(vetoed & set(self.db.approved_prs(repo.slug)))
        if only:
            contradicted = [pr for pr in contradicted if pr in only]
        logger.info(
            "%s: %d PR(s) carry a rejection of mine, %d of them contradicted by a later ok",
            repo.slug, len(vetoed), len(contradicted),
        )
        out: list[Outcome] = []
        for pr in contradicted:
            try:
                detail = await self._dismiss_if_vetoed(repo, pr)
            except GhError as ex:
                logger.warning("%s#%s: could not dismiss: %s", repo.slug, pr, ex)
                out.append(Outcome(repo.slug, pr, "failed", str(ex)[:120]))
                continue
            out.append(Outcome(repo.slug, pr, "retract", detail or "nothing of mine standing"))
        return out

    async def _retract_rejection(self, repo: RepoConfig, meta: PrMeta) -> None:
        """Take back an earlier changes-requested of ours, now that this pass is an ok.

        Best-effort: the approval and the CI request matter more than the tidy-up,
        and a token that cannot dismiss must not cost the PR its build.
        """
        try:
            detail = await self._dismiss_if_vetoed(repo, meta.number)
        except GhError as ex:
            logger.warning(
                "%s#%s: could not dismiss my own changes-requested: %s",
                repo.slug, meta.number, ex,
            )
            await self._dm_owner_once(
                f"dismiss:{repo.slug}:{meta.number}",
                f"I approved *{meta.title}* ({meta.url}) but couldn't retract my earlier "
                f"changes-requested review, so GitHub still reads the PR as blocked by "
                f"me: {ex}",
            )
            return
        if detail:
            logger.info("%s#%s: %s", repo.slug, meta.number, detail)

    async def _request_ci(self, repo: RepoConfig, meta: PrMeta) -> str:
        """Trigger a build for an approved commit, at most once per commit: CI does
        not run on push, so a build is something robbie spends rather than sees.

        Unless the author already paid for one. `once_per` only knows what robbie
        itself asked for, and a build somebody else started is just as good.
        """
        if (started := await self._ci_already_started(repo, meta)) is not None:
            return started
        key = f"run-ci:{repo.slug}:{meta.number}:{meta.head_sha}"
        try:
            result = await publish.once_per(
                self.db, key,
                lambda: publish.request_ci(repo, meta, dry_run=self.no_publish),
                quiet=self.no_publish,
            )
        except GhError as ex:
            logger.warning("could not ask for CI on %s#%s: %s", repo.slug, meta.number, ex)
            await self.slack.dm_owner(
                f"I approved *{meta.title}* ({meta.url}) but couldn't post "
                f"`{repo.ci_phrase}`, so CI has not started: {ex}"
            )
            return "CI request failed"
        if result is None:
            return f"CI already asked for {meta.head_sha[:8]}"
        return result.detail

    async def _ci_already_started(self, repo: RepoConfig, meta: PrMeta) -> str | None:
        """The build the author asked for while the review was still running, if any.

        Read fresh: the meta this review ran on was fetched before the container
        started, which for a long review is half an hour of someone else's clicks.

        Every uncertain answer asks anyway. A duplicate build costs minutes of CI;
        a build nobody asked for costs the approval its verdict, and the author
        finds out when the PR has sat green-less for a day.
        """
        try:
            # not under `_gate_sem`, like the rest of the publish path: a read taken
            # after a container has finished must not queue the next tick's gates
            fresh = await pr_meta(repo.slug, meta.number)
        except GhError as ex:
            logger.warning("%s#%s: could not check for a running build: %s",
                           repo.slug, meta.number, ex)
            return None
        if fresh.head_sha != meta.head_sha or not ci_started(fresh, ignore=repo.ignore_checks):
            return None
        return f"CI already running for {meta.head_sha[:8]}"

    async def _token_login(self) -> str | None:
        if self._self_login is None:
            try:
                self._self_login = await whoami()
            except GhError:
                logger.warning("could not resolve the token's own login")
                self._self_login = ""
        return self._self_login or None

    # ----- notifications -------------------------------------------------

    async def _notify(
        self, repo: RepoConfig, meta: PrMeta, verdict: str, findings: list[dict]
    ) -> None:
        phrase = slackmod.verdict_phrase(
            verdict,
            blocking=severity_count(findings, blocking=True),
            should_fix=severity_count(findings, blocking=False),
        )
        if repo.slack_channel:
            await self.slack.post(
                repo.slack_channel,
                slackmod.channel_note(meta.number, meta.title, meta.url, meta.author, phrase),
            )
        state = await self.slack.dm_author(
            meta.author, slackmod.author_note(meta.number, meta.title, meta.url, phrase)
        )
        if state == "unmapped" and self.db.notice_once(f"nomap:{meta.author}"):
            await self.slack.dm_owner(
                slackmod.unmapped_note(meta.author, self.cfg.slack.users_file)
            )

    async def _announce_approval(self, meta: PrMeta, ci: str) -> None:
        """An `ok` leaves no trace on the PR, so it is the only verdict that has to
        be told — and to everyone whose queue it just left, not only the operator."""
        await self.slack.dm_reviewers(
            slackmod.approved_note(meta.number, meta.title, meta.url, ci)
        )

    # ----- cold start ----------------------------------------------------

    async def _retire_unlabeled(self, repo: RepoConfig) -> None:
        """Drop from the panel whatever no longer carries the review label, and
        whatever has moved past the commit robbie judged.

        The per-PR gates cannot do either: a merged PR, a closed one, or one whose
        label a human removed is not in the queue any more, so nothing walks past
        it again — and `ci_watch` stops revisiting an approval once its build
        reports, so a later push goes unnoticed. This is the only read that sees
        the label without the review request attached, which is why it is a second
        search and not `prs` above; it carries the head shas for the same call.
        """
        try:
            heads = await labeled_heads(repo.slug, label=repo.label)
        except GhError as ex:
            logger.warning("could not read %s's labelled PRs: %s", repo.slug, ex)
            return
        if len(heads) == QUEUE_LIMIT:
            # a truncated page would read as "these PRs lost the label" and retire
            # every row past it
            logger.warning("%s: labelled read filled its page; not retiring", repo.slug)
            return
        if self.dry_run:
            logger.info("DRY would retire %s rows outside %d labelled PR(s)",
                        repo.slug, len(heads))
            return
        if retired := self.db.settle_unlabeled(repo.slug, list(heads)):
            logger.info("%s: retired %d row(s) whose PR no longer carries the label",
                        repo.slug, retired)
        if stale := self.db.settle_stale(
            repo.slug, {pr: row.head for pr, row in heads.items()}
        ):
            logger.info("%s: retired %d approval(s) whose commit is no longer the head",
                        repo.slug, stale)
        # after set_requested above, and reversible on purpose: the label coming off
        # puts the PR back on the board next tick, with no new pass needed
        braked = [
            pr for pr, row in heads.items()
            if any(name in repo.brake_labels for name in row.labels)
        ]
        if dropped := self.db.unrequest(repo.slug, braked):
            logger.info("%s: took %d PR(s) off the board — a brake label is on",
                        repo.slug, dropped)

    async def _seed(self, repo: RepoConfig, prs: list[int]) -> None:
        """Record the current backlog instead of reviewing it.

        Enabling robbie on a repo with 20 pending PRs must not post 20 reviews.
        """
        for pr in prs:
            try:
                meta = await pr_meta(repo.slug, pr)
                requested_at = await last_review_request(repo.slug, pr, repo.reviewer_login)
            except GhError as ex:
                logger.warning("could not seed %s#%s: %s", repo.slug, pr, ex)
                return  # leave the repo unseeded; the next tick tries again
            if not self.dry_run:
                self.db.record_hold(
                    key=dedup_key(repo.slug, pr, meta.head_sha, requested_at),
                    repo=repo.slug, pr=pr, head_sha=meta.head_sha,
                    requested_at=requested_at, reason="backlog at cold start",
                )
        if self.dry_run:
            logger.info("DRY would seed %s with %d PR(s)", repo.slug, len(prs))
            return
        self.db.mark_seeded(repo.slug)
        await self.slack.dm_owner(
            f"robbie is on for *{repo.slug}*. {len(prs)} PR(s) currently request "
            f"{repo.reviewer_login}'s review with the \"{repo.label}\" label; I've noted them "
            "and will review *new* requests from here on. Run "
            f"`robbie once --repo {repo.slug} --pr <n>` to go through one from the backlog."
        )


class _Unusable(NamedTuple):
    reason: str  # the hold_reason on the row
    told: str  # the middle of the operator's DM
    detail: str  # what the Outcome carries


def _unusable(blocks: Blocks) -> _Unusable | None:
    """Why a finished run cannot be published, if it cannot. Held rather than
    failed: the row keeps no verdict, so the PR comes back on its own."""
    if blocks.verdict is None:
        return _Unusable("run gave no verdict", "couldn't parse a verdict", "no verdict")
    if blocks.verdict != "ok" and not blocks.publishable:
        return _Unusable(
            "verdict without a summary body",
            f"got a {blocks.verdict} verdict with no summary body",
            "no summary body",
        )
    return None


async def _requested_at_or_blank(slug: str, pr: int, reviewer: str) -> str:
    try:
        return await last_review_request(slug, pr, reviewer)
    except GhError:
        return "forced"
