"""The pipeline's state transitions, with the container and GitHub faked out.

What these protect: a failed run must come back next tick, and a published one
must not — including when publishing itself failed, because retrying a permanent
publish error would burn a full review every tick.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robbie import ci_watch as ci_mod
from robbie import orchestrator as orch_mod
from robbie import publish as publish_mod
from robbie.budget import Verdict
from robbie.config import Config, DockerConfig, RepoConfig, ReviewModel, Secrets, SlackConfig
from robbie.contract import Blocks
from robbie.db import Db
from robbie.gates import Decision, already_judged, dedup_key
from robbie.github import GhError, LabeledPr, PrMeta, PrThreads, Thread
from robbie.orchestrator import Orchestrator
from robbie.publish import PublishResult
from robbie.runner import ReviewRun

KEY = dedup_key("acme/app", 7, "abc1234567", "2026-01-01T00:00:00Z")
REQ = "2026-01-01T00:00:00Z"


class FakeSlack:
    def __init__(self) -> None:
        self.owner: list[str] = []
        self.reviewers: list[str] = []
        self.channels: list[tuple[str, str]] = []
        self.authors: list[str] = []
        self.author_state = "sent"

    async def dm_owner(self, text: str) -> bool:
        self.owner.append(text)
        return True

    async def dm_reviewers(self, text: str) -> bool:
        self.reviewers.append(text)
        return True

    async def post(self, channel: str, text: str) -> bool:
        self.channels.append((channel, text))
        return True

    async def dm_author(self, login: str, text: str) -> str:
        self.authors.append(login)
        return self.author_state


@pytest.fixture
def repo() -> RepoConfig:
    return RepoConfig(
        slug="acme/app", reviewer_login="rev", bare=Path("/srv/m/app.git"),
        slack_channel="C0CHAN",
    )


@pytest.fixture
def cfg(tmp_path, repo) -> Config:
    return Config(
        slack=SlackConfig(owner_id="U0OWNER"), repos=[repo], state_dir=tmp_path,
        docker=DockerConfig(timeout_s=5),
    )


@pytest.fixture
def orch(cfg, tmp_path, monkeypatch):
    db = Db(tmp_path / "robbie.db")
    slack = FakeSlack()
    secrets = Secrets(
        gh_token="w", slack_bot_token="s", reviewer_gh_token="r", anthropic_api_key="sk"
    )
    o = Orchestrator(cfg, secrets, db, slack)  # type: ignore[arg-type]
    # nothing in these tests may reach GitHub; the ok path looks for a rejection
    # of ours to retract, and having none is the ordinary case
    monkeypatch.setattr(orch_mod, "standing_rejection", _async(None))
    yield o
    db.close()


def _async(value):
    async def _call(*a, **kw):
        return value
    return _call


def _boom(error):
    async def _call(*a, **kw):
        raise error
    return _call


def pr(**kw) -> PrMeta:
    base = dict(
        number=7, title="Add widgets", url="https://x/7", author="dev",
        head_sha="abc1234567", changed_files=2, labels=(), checks=(),
    )
    return PrMeta(**{**base, **kw})


def stub_run(monkeypatch, run: ReviewRun) -> None:
    monkeypatch.setattr(orch_mod, "run_review", _async(run))


def ok_run(verdict: str, *, inline: str = "[]") -> ReviewRun:
    return ReviewRun(
        ok=True,
        blocks=Blocks(verdict=verdict, github="summary", inline=inline),
        cost_usd=0.42, tokens_in=1000, tokens_out=200, duration_s=12.0,
        transcript=Path("/tmp/t.md"),
    )


async def test_the_gate_cache_does_not_cross_repos(orch, cfg, monkeypatch):
    """Two repos can each have a PR #7, and the conversation is not shared."""
    other = RepoConfig(slug="acme/other", reviewer_login="rev", bare=Path("/srv/m/o.git"))
    cfg.repos.append(other)

    async def threads(slug, number, reviewer):
        return PrThreads(state="OPEN", threads=[Thread(
            path=f"{slug}#{number}", line=1, resolved=False, outdated=False,
            mine="a finding", replies=(),
        )])

    monkeypatch.setattr(orch_mod, "my_threads", threads)
    monkeypatch.setattr(orch_mod, "pr_meta", _async(pr()))
    monkeypatch.setattr(orch_mod, "last_review_request", _async(REQ))

    await orch._gate(cfg.repos[0], 7)
    prior = await orch._prior_threads(other, pr())
    assert [t.path for t in prior] == ["acme/other#7"]


# ----- a PR a human has taken ----------------------------------------------


def _taken(cfg, monkeypatch, labels):
    """A PR carrying `labels`, with the timeline read wired to fail if reached."""
    cfg.repos[0] = cfg.repos[0].model_copy(
        update={"hold_labels": ("Blocked",), "done_labels": ("Ready for Prod",)}
    )
    monkeypatch.setattr(orch_mod, "pr_meta", _async(pr(labels=labels)))

    async def boom(*a, **kw):
        raise AssertionError("the timeline read is what this gate exists to skip")

    monkeypatch.setattr(orch_mod, "last_review_request", boom)
    monkeypatch.setattr(orch_mod, "my_threads", boom)
    return cfg.repos[0]


async def test_a_done_label_skips_before_the_expensive_read(orch, cfg, monkeypatch):
    repo = _taken(cfg, monkeypatch, ("Code Review", "Ready for Prod"))
    _, _, _, decision = await orch._gate(repo, 7)
    assert decision.action == "skip"
    assert "a human has it" in decision.reason


async def test_a_hold_label_skips_before_the_expensive_read(orch, cfg, monkeypatch):
    repo = _taken(cfg, monkeypatch, ("Code Review", "Blocked"))
    _, _, _, decision = await orch._gate(repo, 7)
    assert decision.action == "hold"
    assert decision.record is False, "it comes back when the dependency merges"


async def test_a_done_label_retires_the_pr_from_the_panel_and_the_sweep(
    orch, cfg, monkeypatch
):
    repo = _taken(cfg, monkeypatch, ("Code Review", "Ready for Prod"))
    orch.db.start_review(
        key=KEY, repo="acme/app", pr=7, head_sha="abc1234567", requested_at=REQ
    )
    orch.db.finish_review(KEY, state="published", verdict="ok", ci_state="waiting", inline=1)
    assert orch.db.reviewed_prs("acme/app") == [7]

    await orch._gate(repo, 7)

    assert orch.db.reviewed_prs("acme/app") == [], "the reply sweep stops paying for it"
    assert orch.db.approved_and_green(0) == [], "and it leaves the dashboard"


# ----- a PR that left the label behind -------------------------------------


def _labeled(monkeypatch, prs, boom=False, head="abc1234567", labels=()):
    async def read(slug, *, label):
        if boom:
            raise GhError("gh exploded")
        return {
            int(pr): LabeledPr(head=head, labels=("Code Review", *labels)) for pr in prs
        }

    monkeypatch.setattr(orch_mod, "labeled_heads", read)


def _approved(db, pr_number=7):
    key = dedup_key("acme/app", pr_number, "abc1234567", REQ)
    db.start_review(
        key=key, repo="acme/app", pr=pr_number, head_sha="abc1234567", requested_at=REQ
    )
    # inline=1: threads of ours can exist, so the reply sweep keeps the PR
    db.finish_review(key, state="published", verdict="ok", ci_state="green", inline=1)
    # the panel's second condition: somebody is still asking for the review
    db.set_requested("acme/app", [pr_number])


async def test_a_pr_without_the_label_leaves_the_panel(orch, cfg, monkeypatch):
    """Merged, closed, or the label taken off — none of them reach the gates."""
    _approved(orch.db)
    _labeled(monkeypatch, [])

    await orch._retire_unlabeled(cfg.repos[0])

    assert orch.db.approved_and_green(0) == []
    assert orch.db.reviewed_prs("acme/app") == [7], "but the reply sweep keeps it"


async def test_a_pr_still_carrying_the_label_stays(orch, cfg, monkeypatch):
    _approved(orch.db)
    _labeled(monkeypatch, [7])

    await orch._retire_unlabeled(cfg.repos[0])

    assert [r["pr"] for r in orch.db.approved_and_green(0)] == [7]


async def test_a_brake_label_takes_the_pr_off_the_board(orch, cfg, monkeypatch):
    """The complaint this fixes: a row saying "ready for a human" that opens onto a
    needs-work label."""
    _approved(orch.db)
    _labeled(monkeypatch, [7], labels=("❌ NEEDS WORK! ❌",))

    await orch._retire_unlabeled(cfg.repos[0])

    assert orch.db.approved_and_green(0) == []


async def test_the_label_coming_off_puts_it_back_without_a_new_pass(orch, cfg, monkeypatch):
    _approved(orch.db)
    _labeled(monkeypatch, [7], labels=("❌ NEEDS WORK! ❌",))
    await orch._retire_unlabeled(cfg.repos[0])
    assert orch.db.approved_and_green(0) == []

    # next tick: the queue read repopulates, and the label is gone
    orch.db.set_requested("acme/app", [7])
    _labeled(monkeypatch, [7])
    await orch._retire_unlabeled(cfg.repos[0])

    assert [r["pr"] for r in orch.db.approved_and_green(0)] == [7]


async def test_a_moved_head_stays_off_even_with_no_brake_label(orch, cfg, monkeypatch):
    """The other half of the rule: new code waits for a pass, label or no label."""
    _approved(orch.db)
    _labeled(monkeypatch, [7], head="9999999999")

    await orch._retire_unlabeled(cfg.repos[0])
    orch.db.set_requested("acme/app", [7])

    assert orch.db.approved_and_green(0) == [], "reviewed at abc1234567, head is 9999999999"


async def test_a_hold_label_counts_as_a_brake(orch, cfg, monkeypatch):
    repo = cfg.repos[0].model_copy(update={"hold_labels": ("Blocked",)})
    _approved(orch.db)
    _labeled(monkeypatch, [7], labels=("Blocked",))

    await orch._retire_unlabeled(repo)

    assert orch.db.approved_and_green(0) == []


async def test_a_failed_label_read_retires_nothing(orch, cfg, monkeypatch):
    """Every PR looks unlabelled when the read is what broke."""
    _approved(orch.db)
    _labeled(monkeypatch, [], boom=True)

    await orch._retire_unlabeled(cfg.repos[0])

    assert [r["pr"] for r in orch.db.approved_and_green(0)] == [7]


async def test_a_truncated_label_read_retires_nothing(orch, cfg, monkeypatch):
    _approved(orch.db, pr_number=99_999)
    _labeled(monkeypatch, range(1, orch_mod.QUEUE_LIMIT + 1))

    await orch._retire_unlabeled(cfg.repos[0])

    assert [r["pr"] for r in orch.db.approved_and_green(0)] == [99_999]


async def test_a_dry_run_retires_nothing(orch, cfg, monkeypatch):
    _approved(orch.db)
    _labeled(monkeypatch, [])
    orch.dry_run = True

    await orch._retire_unlabeled(cfg.repos[0])

    assert [r["pr"] for r in orch.db.approved_and_green(0)] == [7]


async def test_a_dry_run_marks_nothing(orch, cfg, monkeypatch):
    repo = _taken(cfg, monkeypatch, ("Code Review", "Ready for Prod"))
    orch.dry_run = True
    orch.db.start_review(
        key=KEY, repo="acme/app", pr=7, head_sha="abc1234567", requested_at=REQ
    )
    orch.db.finish_review(KEY, state="published", verdict="ok", ci_state="waiting", inline=1)

    await orch._gate(repo, 7)

    assert orch.db.reviewed_prs("acme/app") == [7]


# ----- which arm reviews, and what covers for it ---------------------------


def _arms(cfg) -> None:
    cfg.review_models = [
        ReviewModel(model="glm-5.2:cloud", via="endpoint", weight=2),
        ReviewModel(model="sonnet", weight=1),
    ]


def _meters(monkeypatch, *, account: bool, endpoint: bool) -> list[bool]:
    """Answer each arm's gate independently, and record which was asked."""
    asked: list[bool] = []

    def gate(cfg, secrets, db, inflight=0, *, via_endpoint=False):
        asked.append(via_endpoint)
        ok = endpoint if via_endpoint else account
        return Verdict(ok, "endpoint" if via_endpoint else "account")

    monkeypatch.setattr(orch_mod.budget, "check", gate)
    return asked


async def test_a_spent_account_falls_back_to_the_endpoint(orch, cfg, monkeypatch):
    """Today's live case: the plan window full while the endpoint sits at 0.1%."""
    _arms(cfg)
    _meters(monkeypatch, account=False, endpoint=True)
    key = next(k for k in (f"k{n}" for n in range(50)) if not cfg.choose_model(k).via_endpoint)
    choice, verdict = await orch._admit(key)
    assert verdict.allowed
    assert (choice.model, choice.via_endpoint) == ("glm-5.2:cloud", True)


async def test_a_spent_endpoint_falls_back_to_the_account(orch, cfg, monkeypatch):
    _arms(cfg)
    _meters(monkeypatch, account=True, endpoint=False)
    key = next(k for k in (f"k{n}" for n in range(50)) if cfg.choose_model(k).via_endpoint)
    choice, verdict = await orch._admit(key)
    assert verdict.allowed
    assert (choice.model, choice.via_endpoint) == ("sonnet", False)


async def test_both_spent_refuses_with_the_reason_of_the_arm_it_wanted(orch, cfg, monkeypatch):
    _arms(cfg)
    _meters(monkeypatch, account=False, endpoint=False)
    choice, verdict = await orch._admit("k1")
    assert not verdict.allowed
    assert verdict.detail == ("endpoint" if choice.via_endpoint else "account")


async def test_a_room_to_spare_arm_is_never_asked(orch, cfg, monkeypatch):
    _arms(cfg)
    asked = _meters(monkeypatch, account=True, endpoint=True)
    await orch._admit("k1")
    assert len(asked) == 1, "the second meter is only read when the first says no"


async def test_nothing_configured_has_nothing_to_fall_back_to(orch, monkeypatch):
    _meters(monkeypatch, account=False, endpoint=True)
    choice, verdict = await orch._admit("k1")
    assert not verdict.allowed
    assert choice.model is None, "the account's own model is the only arm there is"


async def test_a_model_named_on_the_cli_is_never_substituted(orch, cfg, monkeypatch):
    """An explicit --model is a request, not a routing preference."""
    _arms(cfg)
    _meters(monkeypatch, account=True, endpoint=False)
    orch.model = "glm-5.2:cloud"
    choice, verdict = await orch._admit("k1")
    assert not verdict.allowed
    assert choice.model == "glm-5.2:cloud"


# ----- holds ---------------------------------------------------------------


async def test_a_recorded_hold_dms_once_and_is_not_looked_at_again(orch, repo):
    decision = Decision("hold", "nothing new pushed", dm="heads up", record=True)
    await orch._act(repo, pr(), KEY, REQ, decision)
    await orch._act(repo, pr(), KEY, REQ, decision)
    assert orch.slack.owner == ["heads up"], "one DM per key, not one per tick"
    assert orch.db.get_review(KEY).state == "held"


async def test_a_retryable_hold_leaves_no_row(orch, repo):
    await orch._act(repo, pr(), KEY, REQ, Decision("hold", "comments open", dm="x", record=False))
    assert orch.db.get_review(KEY) is None, "the next tick has to look again"
    assert orch.slack.owner == ["x"]


async def test_a_silent_hold_never_dms(orch, repo):
    await orch._act(repo, pr(), KEY, REQ, Decision("hold", "label on", record=False))
    assert orch.slack.owner == []


async def test_a_dry_run_does_not_spend_the_one_shot_dm(orch, repo):
    """--dry-run is how a deployment is proved before it reviews anything.

    Recording the notice there would make the operator DM for every held PR
    vanish from the next real tick, which is the opposite of writing nothing.
    """
    decision = Decision("hold", "nothing new pushed", dm="they asked again")
    orch.dry_run = True
    await orch._act(repo, pr(), KEY, REQ, decision)
    assert orch.slack.owner == ["they asked again"], "a dry run still says what it would send"

    orch.dry_run = False
    await orch._act(repo, pr(), KEY, REQ, decision)
    assert orch.slack.owner == ["they asked again"] * 2, "the real tick still owes the DM"

    await orch._act(repo, pr(), KEY, REQ, decision)
    assert len(orch.slack.owner) == 2, "and owes it once, which is what the key is for"


# ----- a failed run --------------------------------------------------------


async def test_a_failed_run_is_retryable_and_reported(orch, repo, monkeypatch):
    stub_run(monkeypatch, ReviewRun(ok=False, error="container exited 1", duration_s=3.0))
    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert outcome.action == "failed"
    assert orch.db.get_review(KEY).state == "failed"
    assert not orch.db.sha_was_judged("acme/app", 7, "abc1234567")
    assert "the run failed" in orch.slack.owner[0]


async def test_a_run_without_a_verdict_posts_nothing_and_stops_retrying(orch, repo, monkeypatch):
    stub_run(monkeypatch, ReviewRun(ok=True, blocks=Blocks(None, "", ""), duration_s=1.0))
    called = []
    monkeypatch.setattr(publish_mod, "publish_review", lambda *a, **k: called.append(1))
    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert outcome.action == "failed"
    assert called == [], "no verdict means nothing gets posted"
    assert orch.db.get_review(KEY).state == "held"
    assert orch.db.sha_was_judged("acme/app", 7, "abc1234567"), "a 30-min run is not retried blind"


async def test_a_verdict_with_no_summary_body_posts_nothing_either(orch, repo, monkeypatch):
    stub_run(monkeypatch, ReviewRun(
        ok=True, blocks=Blocks(verdict="needs-work", github="  ", inline="[]"), duration_s=1.0
    ))
    called = []
    monkeypatch.setattr(publish_mod, "publish_review", lambda *a, **k: called.append(1))
    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert outcome.action == "failed" and outcome.detail == "no summary body"
    assert called == [], "a review with nothing to head it is not posted"
    assert orch.db.get_review(KEY).state == "held"
    assert "needs-work" in orch.slack.owner[0], "the DM says which verdict was lost"


# ----- verdicts ------------------------------------------------------------


def _build(monkeypatch, *checks) -> None:
    """What a fresh read of the PR finds on the head commit, if anything."""
    monkeypatch.setattr(orch_mod, "pr_meta", _async(pr(checks=checks)))


async def test_retract_only_touches_prs_a_later_ok_contradicted(orch, cfg, monkeypatch):
    """A PR whose newest pass is still a needs-work keeps its veto: that one is true."""
    for pr_num, verdict in ((7, "ok"), (8, "needs-work"), (9, "ok")):
        key = dedup_key("acme/app", pr_num, "abc1234567", REQ)
        orch.db.start_review(key=key, repo="acme/app", pr=pr_num,
                             head_sha="abc1234567", requested_at=REQ)
        orch.db.finish_review(key, state="published", verdict=verdict)
    # 9 was approved but carries no rejection; 10 was never reviewed here
    monkeypatch.setattr(
        orch_mod, "stale_changes_requested",
        _async([{"number": 7}, {"number": 8}, {"number": 10}]),
    )
    dismissed = []

    async def rejection(slug, pr, reviewer):
        return f"PRR_{pr}"

    monkeypatch.setattr(orch_mod, "standing_rejection", rejection)
    monkeypatch.setattr(
        publish_mod, "dismiss_own_rejection",
        lambda node_id, **k: _mark(dismissed, PublishResult(True, f"dismissed {node_id}")),
    )

    outcomes = await orch.retract_stale_rejections(cfg.repos[0])

    assert [o.pr for o in outcomes] == [7], "8 is still rejected, 10 we never approved"
    assert len(dismissed) == 1


async def test_retract_reports_a_denial_instead_of_dying(orch, cfg, monkeypatch):
    key = dedup_key("acme/app", 7, "abc1234567", REQ)
    orch.db.start_review(key=key, repo="acme/app", pr=7,
                         head_sha="abc1234567", requested_at=REQ)
    orch.db.finish_review(key, state="published", verdict="ok")
    monkeypatch.setattr(orch_mod, "stale_changes_requested", _async([{"number": 7}]))
    monkeypatch.setattr(orch_mod, "standing_rejection", _async("PRR_7"))
    monkeypatch.setattr(
        publish_mod, "dismiss_own_rejection", _boom(GhError("HTTP 403: not accessible")),
    )

    outcomes = await orch.retract_stale_rejections(cfg.repos[0])

    assert [(o.pr, o.action) for o in outcomes] == [(7, "failed")]


async def test_ok_retracts_an_earlier_rejection_of_ours(orch, repo, monkeypatch):
    """Otherwise GitHub keeps reporting the PR as blocked by us after we said it
    was fine, and every consumer of `reviewDecision` believes it."""
    _build(monkeypatch)
    dismissed = []
    monkeypatch.setattr(orch_mod, "standing_rejection", _async("PRR_node1"))
    monkeypatch.setattr(
        publish_mod, "dismiss_own_rejection",
        lambda node_id, **k: _mark(dismissed, PublishResult(True, f"dismissed {node_id}")),
    )
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(publish_mod, "request_ci", _async(PublishResult(True, "ci")))
    stub_run(monkeypatch, ok_run("ok"))

    await orch._review(repo, pr(), KEY, REQ)
    assert dismissed, "an ok has to take back our own changes-requested"


async def test_a_token_that_cannot_dismiss_still_gets_the_build(orch, repo, monkeypatch):
    """The tidy-up is best-effort; the approval and the build are the point."""
    _build(monkeypatch)
    ci = []
    monkeypatch.setattr(orch_mod, "standing_rejection", _async("PRR_node1"))
    monkeypatch.setattr(
        publish_mod, "dismiss_own_rejection",
        _boom(GhError("HTTP 403: Resource not accessible by personal access token")),
    )
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(
        publish_mod, "request_ci",
        lambda *a, **k: _mark(ci, PublishResult(True, "asked CI to run (run-ci)")),
    )
    stub_run(monkeypatch, ok_run("ok"))

    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert ci, "a token that cannot dismiss must not cost the PR its build"
    assert orch.db.get_review(KEY).verdict == "ok"
    assert outcome.action == "review"
    assert any("blocked by" in text for text in orch.slack.owner), "the operator hears it"


async def test_ok_clears_the_label_asks_for_ci_and_says_so(orch, repo, monkeypatch):
    _build(monkeypatch)
    cleared, ci = [], []
    monkeypatch.setattr(
        publish_mod, "clear_needs_work",
        lambda *a, **k: _mark(cleared, PublishResult(True, "cleared")),
    )
    monkeypatch.setattr(
        publish_mod, "request_ci",
        lambda *a, **k: _mark(ci, PublishResult(True, "asked CI to run (run-ci)")),
    )
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "nope")))
    stub_run(monkeypatch, ok_run("ok"))

    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert outcome.action == "review"
    assert cleared, "an ok verdict releases the brake"
    assert ci, "an approval is what pays for a build now that push does not"
    assert orch.db.get_review(KEY).verdict == "ok"
    assert orch.slack.reviewers == [
        "✅ <https://x/7|*#7*> Add widgets — nothing to fix, asked CI to run (run-ci)."
    ], "an ok is invisible on the PR, so one line has to say it happened"
    assert orch.slack.owner == [], "not an operator alert; it goes to whoever reviews"
    assert orch.slack.channels == [], "no review was posted, so nothing to announce"


async def test_an_approval_says_it_opened_no_threads(orch, repo, monkeypatch):
    """The sweep reads this column to skip the PR, and only a recorded 0 means
    there is nothing there — an ok that left it unwritten would be swept forever."""
    _build(monkeypatch)
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(publish_mod, "request_ci", _async(PublishResult(True, "ci")))
    stub_run(monkeypatch, ok_run("ok"))

    await orch._review(repo, pr(), KEY, REQ)
    assert orch.db.reviewed_prs("acme/app") == []


async def test_ci_is_asked_for_once_per_commit(orch, repo, monkeypatch):
    """A build costs money now, and a forced re-review must not buy a second one."""
    _build(monkeypatch)
    ci = []
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(
        publish_mod, "request_ci",
        lambda *a, **k: _mark(ci, PublishResult(True, "asked CI to run (run-ci)")),
    )
    stub_run(monkeypatch, ok_run("ok"))

    first = await orch._review(repo, pr(), KEY, REQ)
    second = await orch._review(repo, pr(), "forced-again", REQ)
    assert len(ci) == 1
    assert "asked CI" in first.detail and "already asked" in second.detail


async def test_a_ci_request_that_failed_is_asked_for_again(orch, repo, monkeypatch):
    """The guard means "the phrase is on the PR", so a failed post must not set it.

    Otherwise an approval nobody builds can never be recovered from: every later
    pass on that commit reads "already asked" and stays silent.
    """
    _build(monkeypatch)
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    stub_run(monkeypatch, ok_run("ok"))

    async def boom(*a, **k):
        raise GhError("502 from github")

    monkeypatch.setattr(publish_mod, "request_ci", boom)
    first = await orch._review(repo, pr(), KEY, REQ)
    assert "CI request failed" in first.detail
    assert any("couldn't post" in m for m in orch.slack.owner)

    ci = []
    monkeypatch.setattr(
        publish_mod, "request_ci", lambda *a, **k: _mark(ci, PublishResult(True, "asked")),
    )
    second = await orch._review(repo, pr(), "forced-again", REQ)
    assert len(ci) == 1 and "asked" in second.detail


async def test_a_build_the_author_already_started_is_not_asked_for_again(
    orch, repo, monkeypatch
):
    """`once_per` only knows what robbie asked for. The author asking first is the
    common case: the review takes half an hour and they are waiting on the same
    build."""
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(
        publish_mod, "request_ci", lambda *a, **k: pytest.fail("asked for a second build"),
    )
    _build(monkeypatch, {"context": "ci/circleci: build", "state": "PENDING"})
    stub_run(monkeypatch, ok_run("ok"))

    outcome = await orch._review(repo, pr(), KEY, REQ)

    assert "CI already running" in outcome.detail
    assert orch.db.get_review(KEY).verdict == "ok"
    orch.db.set_requested("acme/app", [7])  # the panel's other condition
    row = orch.db.approved_and_green(0)[0]
    assert row["ci_state"] == "waiting", "the watch has to report on somebody else's build"


async def test_a_review_bot_ticking_the_commit_is_not_a_build(orch, repo, monkeypatch):
    """It posts a status on every push, so counting it would mean never asking."""
    ci = []
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(
        publish_mod, "request_ci", lambda *a, **k: _mark(ci, PublishResult(True, "asked")),
    )
    _build(monkeypatch, {"context": "CodeRabbit", "state": "SUCCESS"})
    stub_run(monkeypatch, ok_run("ok"))

    await orch._review(repo, pr(), KEY, REQ)

    assert len(ci) == 1


async def test_a_read_that_fails_asks_for_the_build_anyway(orch, repo, monkeypatch):
    """A duplicate build costs minutes; a missing one costs the approval its verdict."""
    ci = []
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(
        publish_mod, "request_ci", lambda *a, **k: _mark(ci, PublishResult(True, "asked")),
    )

    async def boom(*a, **kw):
        raise GhError("502 from github")

    monkeypatch.setattr(orch_mod, "pr_meta", boom)
    stub_run(monkeypatch, ok_run("ok"))

    await orch._review(repo, pr(), KEY, REQ)

    assert len(ci) == 1


async def test_the_ci_trigger_comment_is_the_bare_phrase(repo, monkeypatch):
    """Whatever listens for it may match the whole body, so nothing rides along."""
    sent = {}

    async def fake_gh(*args, stdin=None, **kw):
        sent["body"] = json.loads(stdin)["body"]
        return "https://x/1"

    monkeypatch.setattr(publish_mod, "gh", fake_gh)
    await publish_mod.request_ci(repo, pr())
    assert sent["body"] == "run-ci"


async def test_needs_work_publishes_notifies_and_does_not_brief_the_owner(
    orch, repo, monkeypatch
):
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "did it")))
    stub_run(monkeypatch, ok_run(
        "needs-work", inline='[{"path":"a.rb","line":1,"severity":"Must-fix"}]'
    ))

    await orch._review(repo, pr(), KEY, REQ)
    assert orch.db.get_review(KEY).state == "published"
    assert orch.slack.channels[0][0] == "C0CHAN"
    assert "1 thing to fix" in orch.slack.channels[0][1]
    assert orch.slack.authors == ["dev"]
    assert orch.slack.owner == [], "the PR left their queue; no briefing needed"


async def test_comment_publishes_without_briefing_the_owner(orch, repo, monkeypatch):
    """A published review announces itself on the PR and in the channel."""
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "did it")))
    stub_run(monkeypatch, ok_run("comment"))

    await orch._review(repo, pr(), KEY, REQ)
    assert orch.db.get_review(KEY).verdict == "comment"
    assert orch.slack.authors == ["dev"]
    assert orch.slack.owner == [], "no summary DM for a review that is visible"


async def test_an_unmapped_author_warns_the_owner_once(orch, repo, monkeypatch):
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "did it")))
    stub_run(monkeypatch, ok_run("needs-work"))
    orch.slack.author_state = "unmapped"

    await orch._review(repo, pr(), KEY, REQ)
    await orch._review(repo, pr(), "another-key", REQ)
    assert sum("no Slack id" in m for m in orch.slack.owner) == 1


async def test_nothing_is_announced_when_the_publish_was_a_no_op(orch, repo, monkeypatch):
    monkeypatch.setattr(
        publish_mod, "publish_review", _async(PublishResult(False, "already posted"))
    )
    stub_run(monkeypatch, ok_run("needs-work"))
    await orch._review(repo, pr(), KEY, REQ)
    assert orch.slack.channels == []
    assert orch.slack.authors == []


async def test_a_publish_crash_still_counts_as_judged(orch, repo, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("422 unprocessable")

    monkeypatch.setattr(publish_mod, "publish_review", boom)
    stub_run(monkeypatch, ok_run("needs-work"))

    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert outcome.action == "failed"
    assert orch.db.get_review(KEY).state == "published", (
        "retrying a permanent publish failure would burn a full review every tick"
    )
    assert "post by hand" in orch.slack.owner[0]


async def test_two_models_on_one_commit_keep_their_own_rows(orch, cfg, monkeypatch):
    """Otherwise the second run replaces the first and the comparison has one arm."""
    _arms(cfg)
    monkeypatch.setattr(orch_mod, "pr_meta", _async(pr()))
    monkeypatch.setattr(orch_mod, "last_review_request", _async(REQ))
    monkeypatch.setattr(orch_mod, "my_threads", _async(PrThreads()))
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(False, "-")))
    monkeypatch.setattr(publish_mod, "request_ci", _async(PublishResult(False, "-")))
    stub_run(monkeypatch, ok_run("ok"))

    for model in ("glm-5.2:cloud", "sonnet"):
        orch.model = model
        await orch.review_one("acme/app", 7)

    ran = orch.db.conn.execute(
        "SELECT model FROM reviews WHERE head_sha='abc1234567' ORDER BY model"
    ).fetchall()
    assert [r["model"] for r in ran] == ["glm-5.2:cloud", "sonnet"]


async def test_what_each_model_found_is_recorded_for_the_comparison(orch, repo, monkeypatch):
    """Verdict alone cannot compare models; how much they found at what severity can."""
    inline = json.dumps([
        {"path": "a.rb", "line": 1, "severity": "Critical", "title": "t", "body": "b"},
        {"path": "a.rb", "line": 2, "severity": "Must-fix", "title": "t", "body": "b"},
        {"path": "a.rb", "line": 3, "severity": "Should-fix", "title": "t", "body": "b"},
        {"path": "a.rb", "line": 4, "severity": "Nitpick", "title": "t", "body": "b"},
    ])
    monkeypatch.setattr(
        publish_mod, "publish_review",
        _async(PublishResult(True, "posted", inline=3)),
    )
    stub_run(monkeypatch, ok_run("needs-work", inline=inline))
    await orch._review(repo, pr(), KEY, REQ)

    row = orch.db.conn.execute(
        "SELECT findings, blocking, should_fix, inline, model FROM reviews WHERE key=?", (KEY,)
    ).fetchone()
    assert (row["findings"], row["blocking"], row["should_fix"]) == (4, 2, 1)
    assert row["inline"] == 3, "one of the four could not be anchored to the diff"


async def test_cost_and_tokens_are_recorded(orch, repo, monkeypatch):
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "did it")))
    stub_run(monkeypatch, ok_run("comment"))
    await orch._review(repo, pr(), KEY, REQ)
    assert orch.db.spend_since(0) == pytest.approx(0.42)


# ----- the budget gate ----------------------------------------------------


async def test_the_budget_gate_stops_reviews_and_warns_once(orch, repo, monkeypatch):
    monkeypatch.setattr(orch.cfg.budget, "daily_usd", 0.0)
    monkeypatch.setattr(orch_mod, "my_threads", _async(PrThreads()))

    async def boom(*a, **k):
        raise AssertionError("no container may spawn while the budget is closed")

    monkeypatch.setattr(orch_mod, "run_review", boom)

    for _ in range(3):
        outcome = await orch._review(repo, pr(), KEY, REQ)
    assert outcome.action == "budget"
    assert sum("Holding off" in m for m in orch.slack.owner) == 1


async def test_the_spend_gate_is_asked_once_per_review(orch, repo, monkeypatch):
    """It used to be asked in `_act` and again in `_slot`, and only the first of
    the two ever told the operator anything."""
    asked: list[int] = []
    real = orch_mod.budget.check
    monkeypatch.setattr(
        orch_mod.budget, "check",
        lambda *a, **k: (asked.append(1), real(*a, **k))[1],
    )
    monkeypatch.setattr(orch_mod, "my_threads", _async(PrThreads()))
    stub_run(monkeypatch, ReviewRun(ok=False, error="container exited 1"))

    await orch._act(repo, pr(), KEY, REQ, Decision("review"))
    assert len(asked) == 1


async def test_a_meter_nobody_can_read_is_reported_whichever_meter_it_was(
    orch, repo, monkeypatch
):
    """An unguarded run is unguarded whoever was supposed to be measuring it, and
    the key is dated so the second outage is not the silent one."""
    for key in ("budget:unreadable:2026-08-06", "budget:endpoint-unreadable:2026-08-06"):
        monkeypatch.setattr(
            orch_mod.budget, "check",
            lambda *a, k=key, **kw: Verdict(True, "endpoint down", notice_key=k),
        )
        async with orch._slot(KEY) as slot:
            assert slot.ok, "unreadable is not closed; it runs and says so"
    assert sum("can't read the spend budget" in m for m in orch.slack.owner) == 2


# ----- dry run -----------------------------------------------------------


async def test_no_publish_runs_the_review_but_writes_nothing_outward(orch, repo, monkeypatch):
    orch.no_publish = True
    seen: dict = {}

    async def spy(*a, **kw):
        seen.update(kw)
        return PublishResult(False, "dry run")

    monkeypatch.setattr(publish_mod, "publish_review", spy)
    ran = []
    stub_run(monkeypatch, ok_run("needs-work"))
    monkeypatch.setattr(
        orch_mod, "run_review",
        lambda *a, **kw: ran.append(1) or _async(ok_run("needs-work"))(),
    )

    outcome = await orch._review(repo, pr(), KEY, REQ)
    assert ran == [1], "the review itself must actually run, unlike --dry-run"
    assert seen.get("dry_run") is True, "publishing must be suppressed"
    assert orch.slack.channels == [] and orch.slack.authors == []
    assert outcome.action == "review"


async def test_no_publish_still_records_the_cost(orch, repo, monkeypatch):
    orch.no_publish = True
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(False, "dry run")))
    stub_run(monkeypatch, ok_run("needs-work"))
    await orch._review(repo, pr(), KEY, REQ)
    assert orch.db.spend_since(0) == pytest.approx(0.42), "the run cost real money"


async def test_a_failure_outlives_the_retry_that_replaces_its_row(orch, repo, monkeypatch):
    """The retry reuses the key, so `INSERT OR REPLACE` erases the failed row. If
    nothing else recorded it, the operator DM about it correlates with nothing an
    hour later — which is exactly how a stale one costs an afternoon."""
    stub_run(monkeypatch, ReviewRun(ok=False, error="boom", cost_usd=0.30, duration_s=12.0))
    await orch._review(repo, pr(), KEY, REQ)

    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "posted")))
    stub_run(monkeypatch, ok_run("needs-work"))
    await orch._review(repo, pr(), KEY, REQ)

    assert orch.db.get_review(KEY).state == "published", "the row is the retry's now"
    kept = list(orch.db.conn.execute("SELECT kind, cost_usd FROM spend WHERE pr=7"))
    assert [r["kind"] for r in kept] == ["failed"]
    assert kept[0]["cost_usd"] == pytest.approx(0.30), "a container that died still spent"


async def test_a_third_party_failure_is_not_recorded_as_the_account_s_money(
    orch, cfg, repo, monkeypatch
):
    """`spend` carries no model, so `spend_since` cannot exclude an endpoint arm
    from it the way it does for `reviews`."""
    monkeypatch.setattr(
        cfg, "review_models", [ReviewModel(model="glm-5.2:cloud", via="endpoint")]
    )
    stub_run(monkeypatch, ReviewRun(ok=False, error="boom", cost_usd=9.90, duration_s=12.0))
    await orch._review(repo, pr(), KEY, REQ)

    kept = list(orch.db.conn.execute("SELECT kind, cost_usd FROM spend WHERE pr=7"))
    assert [r["kind"] for r in kept] == ["failed"], "the failure is still recorded"
    assert kept[0]["cost_usd"] is None, "but not as dollars the account was billed"


async def test_no_publish_leaves_the_key_unjudged(orch, repo, monkeypatch):
    """The row is what every gate reads. Recording this pass as published would
    park the PR out of the queue over a review that reached nobody, and the real
    tick behind it would skip it for good."""
    orch.no_publish = True
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(False, "dry run")))
    stub_run(monkeypatch, ok_run("needs-work"))
    await orch._review(repo, pr(), KEY, REQ)

    row = orch.db.get_review(KEY)
    assert row is not None, "the run happened and cost money; the row has to exist"
    assert already_judged(row.state) is None
    assert not orch.db.sha_was_judged("acme/app", 7, "abc1234567"), "gate 4 reads this too"


async def test_no_publish_records_no_hold_either(orch, repo):
    """Same row, same gate: `record=True` holds are the ones that stick."""
    orch.no_publish = True
    await orch._act(repo, pr(), KEY, REQ, Decision("hold", "nothing new pushed", record=True))
    assert orch.db.get_review(KEY) is None


async def test_a_verdict_it_could_not_use_is_not_judged_under_no_publish(orch, repo, monkeypatch):
    """The held-for-a-bad-contract path writes the same gating row as the rest."""
    orch.no_publish = True
    stub_run(monkeypatch, ReviewRun(ok=True, blocks=Blocks(None, "", ""), cost_usd=0.42))
    await orch._review(repo, pr(), KEY, REQ)
    row = orch.db.get_review(KEY)
    assert row is not None and already_judged(row.state) is None


async def test_dry_run_writes_nothing(orch, repo, monkeypatch):
    orch.dry_run = True
    stub_run(monkeypatch, ok_run("needs-work"))
    await orch._act(repo, pr(), KEY, REQ, Decision("hold", "x", dm=None, record=True))
    await orch._act(repo, pr(), KEY, REQ, Decision("review"))
    assert orch.db.get_review(KEY) is None


def _mark(sink, value):
    sink.append(1)

    async def _noop():
        return value

    return _noop()


# ----- what the build said about an approval --------------------------------


def _approve(orch, *, sha="abc1234567", model="glm-5.2:cloud", key="ok-key") -> str:
    orch.db.start_review(key=key, repo="acme/app", pr=7, head_sha=sha, requested_at=REQ)
    orch.db.finish_review(key, state="published", verdict="ok", model=model, ci_state="waiting")
    return key


def _ci_state(orch, key: str) -> str | None:
    return orch.db.conn.execute(
        "SELECT ci_state FROM reviews WHERE key=?", (key,)
    ).fetchone()["ci_state"]


async def test_a_green_build_after_an_approval_is_ready_for_a_human(orch, monkeypatch):
    key = _approve(orch)
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(checks=({"context": "ci/build", "state": "SUCCESS"},))
    ))
    out = await orch.ci.watch()
    assert [o.action for o in out] == ["ready"]
    assert "glm-5.2:cloud" in out[0].detail
    assert _ci_state(orch, key) == "green"


async def test_a_red_build_after_an_approval_is_reported_on_the_pr(orch, monkeypatch):
    key = _approve(orch)
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(checks=({"name": "rspec", "conclusion": "FAILURE"},))
    ))
    told = []
    monkeypatch.setattr(
        publish_mod, "report_red_build",
        lambda *a, **k: _mark(told, PublishResult(True, "reported the red build (rspec)")),
    )
    out = await orch.ci.watch()
    assert told, "the author is the only one who can act on it"
    assert [o.action for o in out] == ["ci-note"]
    assert _ci_state(orch, key) == "red"


async def test_a_build_still_running_stays_on_the_list(orch, monkeypatch):
    key = _approve(orch)
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(checks=({"name": "rspec", "status": "IN_PROGRESS"},))
    ))
    assert await orch.ci.watch() == []
    assert _ci_state(orch, key) == "waiting", "so the next tick looks again"


async def test_a_push_after_the_approval_makes_the_build_irrelevant(orch, monkeypatch):
    """The red belongs to code nobody approved, so it is not news about the approval."""
    key = _approve(orch, sha="oldsha")
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(head_sha="newsha", checks=({"name": "rspec", "conclusion": "FAILURE"},))
    ))
    told = []
    monkeypatch.setattr(publish_mod, "report_red_build",
                        lambda *a, **k: _mark(told, PublishResult(True, "x")))
    assert await orch.ci.watch() == []
    assert _ci_state(orch, key) == "stale"
    assert told == [], "nothing to say about a commit that moved"


async def test_a_closed_pr_leaves_the_list(orch, monkeypatch):
    key = _approve(orch)
    monkeypatch.setattr(ci_mod, "pr_meta", _async(pr(state="MERGED")))
    assert await orch.ci.watch() == []
    assert _ci_state(orch, key) == "gone"


async def test_an_ok_joins_the_watch_list(orch, repo, monkeypatch):
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(True, "c")))
    monkeypatch.setattr(publish_mod, "request_ci", _async(PublishResult(True, "asked")))
    stub_run(monkeypatch, ok_run("ok"))
    await orch._review(repo, pr(), KEY, REQ)
    assert _ci_state(orch, KEY) == "waiting"


async def test_a_no_publish_ok_waits_for_nothing(orch, repo, monkeypatch):
    """It asked no CI, so there is no build coming to wait for."""
    orch.no_publish = True
    monkeypatch.setattr(publish_mod, "clear_needs_work", _async(PublishResult(False, "-")))
    monkeypatch.setattr(publish_mod, "request_ci", _async(PublishResult(False, "dry")))
    stub_run(monkeypatch, ok_run("ok"))
    await orch._review(repo, pr(), KEY, REQ)
    assert _ci_state(orch, KEY) is None


async def test_a_dry_pass_does_not_consume_the_ci_state(orch, monkeypatch):
    """Same rule as the one-shot DMs: deciding without writing must not settle it."""
    key = _approve(orch)
    orch.dry_run = True
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(checks=({"name": "rspec", "conclusion": "FAILURE"},))
    ))
    told = []
    monkeypatch.setattr(publish_mod, "report_red_build",
                        lambda *a, **k: _mark(told, PublishResult(False, "dry run")))
    await orch.ci.watch()
    assert _ci_state(orch, key) == "waiting", "the real tick still owes the comment"


async def test_a_no_publish_pass_does_not_consume_the_ci_state_either(orch, monkeypatch):
    """It ran the review for real but posted nothing, so the note is still owed."""
    key = _approve(orch)
    orch.no_publish = True
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(checks=({"name": "rspec", "conclusion": "FAILURE"},))
    ))
    monkeypatch.setattr(publish_mod, "report_red_build",
                        lambda *a, **k: _async(PublishResult(False, "dry run"))())
    await orch.ci.watch()
    assert _ci_state(orch, key) == "waiting"


async def test_github_going_down_does_not_lose_the_red_build_note(orch, monkeypatch):
    """It also must not take the tick down: the queue is read after this."""
    key = _approve(orch)
    monkeypatch.setattr(ci_mod, "pr_meta", _async(
        pr(checks=({"name": "rspec", "conclusion": "FAILURE"},))
    ))

    async def boom(*a, **k):
        raise GhError("502 Bad Gateway")

    monkeypatch.setattr(publish_mod, "report_red_build", boom)
    assert await orch.ci.watch() == []
    assert _ci_state(orch, key) == "waiting", "unsettled, so the next tick posts it"


# ----- posting once per commit, without asking GitHub ----------------------
#
# The marker scan this replaced cost a full paginated read of every comment on
# the PR, per call. The trade it makes is real: a comment deleted by hand is not
# reposted, so what a run records has to be exactly what it managed to post.


async def _publish_twice(orch, repo, monkeypatch, result: PublishResult) -> list[int]:
    posts: list[int] = []

    async def publishing(*a, **k):
        posts.append(1)
        return result

    monkeypatch.setattr(publish_mod, "publish_review", publishing)
    stub_run(monkeypatch, ok_run("needs-work"))
    for _ in range(2):
        await orch._review(repo, pr(), KEY, REQ)
    return posts


async def test_a_review_is_published_once_per_commit(orch, repo, monkeypatch):
    posts = await _publish_twice(
        orch, repo, monkeypatch, PublishResult(True, "requested changes")
    )
    assert posts == [1], "the second pass on the same commit must post nothing"


async def test_a_publish_that_failed_can_still_be_retried(orch, repo, monkeypatch):
    """Recording on anything but a real post would bury the findings for good."""
    posts = await _publish_twice(orch, repo, monkeypatch, PublishResult(False, "empty body"))
    assert posts == [1, 1]


async def test_no_publish_records_nothing_so_a_real_tick_still_posts(
    orch, repo, monkeypatch
):
    orch.no_publish = True
    await _publish_twice(orch, repo, monkeypatch, PublishResult(False, "dry run"))
    orch.no_publish = False
    posts = await _publish_twice(
        orch, repo, monkeypatch, PublishResult(True, "requested changes")
    )
    assert posts == [1]


async def test_the_ci_note_goes_out_once_per_commit(orch, repo, monkeypatch):
    notes: list[tuple[str, ...]] = []

    async def note(_repo, _meta, checks, *, dry_run=False):
        notes.append(checks)
        return PublishResult(True, "posted the CI note")

    monkeypatch.setattr(publish_mod, "post_ci_note", note)
    red = Decision("ci-note", "ci red", checks=("ci/build",))
    first = await orch._act(repo, pr(), KEY, REQ, red)
    second = await orch._act(repo, pr(), KEY, REQ, red)

    assert notes == [("ci/build",)]
    assert first.action == second.action == "ci-note"
    assert "already noted" in second.detail


async def test_the_red_build_note_goes_out_once_per_commit(orch, repo, monkeypatch):
    orch.db.start_review(
        key=KEY, repo="acme/app", pr=7, head_sha="abc1234567", requested_at=REQ
    )
    orch.db.finish_review(KEY, state="published", verdict="ok", ci_state="waiting")
    posted: list[int] = []

    async def report(*a, **k):
        posted.append(1)
        return PublishResult(True, "reported the red build")

    monkeypatch.setattr(publish_mod, "report_red_build", report)
    monkeypatch.setattr(
        ci_mod, "pr_meta",
        _async(pr(checks=({"context": "ci/build", "state": "FAILURE"},))),
    )

    await orch.ci.watch()
    orch.db.set_ci_state(KEY, "waiting")  # as a second row on the same commit would
    await orch.ci.watch()
    assert posted == [1]


# ----- the props bridge --------------------------------------------------


def _told(monkeypatch):
    told = []

    async def fake_report(url, pr_num, head, status, verdict=None):
        told.append((status, verdict))

    monkeypatch.setattr(orch_mod.props_bridge, "report", fake_report)
    return told


async def test_the_board_hears_the_review_lifecycle(orch, repo, monkeypatch):
    told = _told(monkeypatch)
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "did it")))
    stub_run(monkeypatch, ok_run(
        "needs-work", inline='[{"path":"a.rb","line":1,"severity":"Must-fix"}]'
    ))
    await orch._review(repo, pr(), KEY, REQ)
    assert told == [("reviewing", None), ("posted", "needs-work")], (
        "the verdict rides along so the board can tell an ok from a needs-work"
    )


async def test_a_failed_run_tells_the_board_skipped(orch, repo, monkeypatch):
    told = _told(monkeypatch)
    stub_run(monkeypatch, ReviewRun(ok=False, error="container exited 1", duration_s=3.0))
    await orch._review(repo, pr(), KEY, REQ)
    assert told == [("reviewing", None), ("skipped", None)]


async def test_a_quiet_run_tells_the_board_nothing(orch, repo, monkeypatch):
    told = _told(monkeypatch)
    orch.no_publish = True
    monkeypatch.setattr(publish_mod, "publish_review", _async(PublishResult(True, "did it")))
    stub_run(monkeypatch, ok_run(
        "needs-work", inline='[{"path":"a.rb","line":1,"severity":"Must-fix"}]'
    ))
    await orch._review(repo, pr(), KEY, REQ)
    assert told == [], "--no-publish posts nothing anywhere, the board included"


async def test_the_heartbeat_carries_the_panel_summary(orch, monkeypatch):
    """One payload per tick: what the 4020 panel would say, for the board's card."""
    beats = []

    async def fake_heartbeat(url, payload):
        beats.append(payload)

    monkeypatch.setattr(orch_mod.props_bridge, "heartbeat", fake_heartbeat)
    orch.cfg.props_url = "http://host.docker.internal:4021"
    _approved(orch.db)
    orch.db.record_spend(repo="acme/app", pr=9, kind="failed", cost_usd=0.3, duration_s=5)
    await orch._heartbeat()
    (beat,) = beats
    assert beat["today"] == {"ok": 1}
    assert beat["failed_today"] == 1
    assert beat["ready"] == [7]
    assert beat["reviewing"] == []
    assert "allowed" in beat["budget"] and beat["tick_s"] == orch.cfg.poll_interval_s


async def test_a_quiet_run_sends_no_heartbeat(orch, monkeypatch):
    beats = []

    async def fake_heartbeat(url, payload):
        beats.append(payload)

    monkeypatch.setattr(orch_mod.props_bridge, "heartbeat", fake_heartbeat)
    orch.cfg.props_url = "http://host.docker.internal:4021"
    orch.no_publish = True
    await orch._heartbeat()
    assert beats == []


# ----- the credentials gate ----------------------------------------------


def _creds_file(orch, tmp_path, expires_in_min: float):
    import json as _json
    import time as _time

    f = tmp_path / "creds.json"
    f.write_text(_json.dumps({"claudeAiOauth": {
        "accessToken": "t",
        "expiresAt": (_time.time() + expires_in_min * 60) * 1000,
    }}))
    orch.cfg.backend = "oauth"  # the gate only applies to the oauth backend
    orch.secrets = orch.secrets.model_copy(update={"claude_credentials": f})
    return f


async def test_an_expired_token_pauses_containers_and_dms_once(
    orch, cfg, tmp_path, monkeypatch
):
    """Seven DMs in an hour for one stale token (2026-09-15). While the proxy
    would refuse every model call, a tick spawns nothing and the operator hears
    about the outage exactly once."""
    _creds_file(orch, tmp_path, expires_in_min=-30)
    stale_ms, exp = orch._creds_expired()
    assert stale_ms > 0 and exp > 0

    await orch._dm_owner_once(f"creds-expired:{exp}", "paused")
    await orch._dm_owner_once(f"creds-expired:{exp}", "paused")
    assert orch.slack.owner == ["paused"], "same token, one DM"

    # a refreshed token mints a new expiresAt: the gate opens and a future
    # outage gets its own single announcement
    _creds_file(orch, tmp_path, expires_in_min=60)
    assert orch._creds_expired() == (0, 0)


async def test_a_fresh_token_leaves_the_tick_alone(orch, tmp_path):
    _creds_file(orch, tmp_path, expires_in_min=120)
    assert orch._creds_expired() == (0, 0)


async def test_the_heartbeat_names_the_pause(orch, tmp_path, monkeypatch):
    beats = []

    async def fake_heartbeat(url, payload):
        beats.append(payload)

    monkeypatch.setattr(orch_mod.props_bridge, "heartbeat", fake_heartbeat)
    orch.cfg.props_url = "http://host.docker.internal:4021"
    _creds_file(orch, tmp_path, expires_in_min=-30)
    await orch._heartbeat()
    assert "credentials expired" in beats[0]["paused"]
