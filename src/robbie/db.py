"""SQLite state.

ponytail: sync sqlite3, single writer process. Calls are sub-ms against a poll
loop, so an async wrapper would buy nothing. If robbie ever runs more than one
orchestrator, this is the thing to move to postgres.

Adding a table is free — `IF NOT EXISTS` runs on every boot. Adding a *column* is
not: the CREATE is skipped on an existing database and nothing notices until a
query mentions it. That needs an explicit ALTER, guarded by `PRAGMA user_version`.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    key          TEXT    NOT NULL UNIQUE,   -- repo:pr:head_sha:requested_at
    repo         TEXT    NOT NULL,
    pr           INTEGER NOT NULL,
    head_sha     TEXT    NOT NULL,
    requested_at TEXT    NOT NULL,
    state        TEXT    NOT NULL
                 CHECK (state IN ('running','published','held','failed')),
    verdict      TEXT,                      -- needs-work | comment | ok
    hold_reason  TEXT,
    cost_usd     REAL,
    tokens_in    INTEGER,
    tokens_out   INTEGER,
    duration_s   REAL,
    transcript   TEXT,
    model        TEXT,                      -- null = whatever the account defaults to
    findings     INTEGER,                   -- what the run reported, by severity
    blocking     INTEGER,                   -- critical + must-fix
    should_fix   INTEGER,
    inline       INTEGER,                   -- of those, anchored to a diff line
    summary_findings INTEGER,               -- indexed in the summary instead
    ci_state     TEXT,             -- waiting|green|red|stale|gone|done|unlabeled|expired
    ci_seen_at   INTEGER,
    created_at   INTEGER NOT NULL,
    finished_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_reviews_pr      ON reviews(repo, pr, head_sha);
CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at DESC);

-- one-shot operator notices, so a hold never DMs twice for the same key
CREATE TABLE IF NOT EXISTS notices (
    key        TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);

-- containers that spend without being a review pass, so the budget can see them
CREATE TABLE IF NOT EXISTS spend (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    repo       TEXT    NOT NULL,
    pr         INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    cost_usd   REAL,
    duration_s REAL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spend_created ON spend(created_at DESC);

-- what a provider last said about its own limits. One poller writes it and
-- every process reads it, so the request rate is the poll interval and not
-- "once per gate, times however many robbie processes are up"
CREATE TABLE IF NOT EXISTS meters (
    name     TEXT PRIMARY KEY,   -- plan | endpoint
    pct      REAL,
    note     TEXT,
    read_at  INTEGER,            -- of the last SUCCESSFUL read; null = never
    error    TEXT,               -- the last failed one, cleared by a success
    error_at INTEGER
);

-- what the last queue read saw still asking for our review. The panel joins
-- against it: a PR nobody has asked about is not something a human is waiting on,
-- and an author who never re-requests is `digest`'s problem, not the board's.
CREATE TABLE IF NOT EXISTS requested (
    repo    TEXT    NOT NULL,
    pr      INTEGER NOT NULL,
    seen_at INTEGER NOT NULL,
    PRIMARY KEY (repo, pr)
);

-- cold start: the first poll of a repo records its backlog instead of
-- reviewing it, so enabling robbie can't trigger a review storm
CREATE TABLE IF NOT EXISTS seeded (
    repo      TEXT PRIMARY KEY,
    seeded_at INTEGER NOT NULL
);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class ReviewRow:
    key: str
    repo: str
    pr: int
    head_sha: str
    state: str
    verdict: str | None
    hold_reason: str | None


SCHEMA_VERSION = 4


class Db:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        if read_only:
            # the dashboard opens this way, so a bug there cannot touch a review's
            # row. The directory still has to be writable: SQLite needs the -shm
            # file to read a WAL database at all.
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            return
        # check_same_thread=False because the spend gate is read through
        # asyncio.to_thread. Safe only while sqlite3.threadsafety is 3 (serialized)
        # and this stays a single writer process.
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Bring an existing database up to what the code above assumes."""
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        added = {
            1: [("model", "TEXT")],
            2: [("findings", "INTEGER"), ("blocking", "INTEGER"),
                ("should_fix", "INTEGER"), ("inline", "INTEGER")],
            3: [("summary_findings", "INTEGER")],
            4: [("ci_state", "TEXT"), ("ci_seen_at", "INTEGER")],
        }
        for step in range(version + 1, SCHEMA_VERSION + 1):
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(reviews)")}
            for name, kind in added.get(step, []):
                if name not in columns:
                    self.conn.execute(f"ALTER TABLE reviews ADD COLUMN {name} {kind}")
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    # ----- reviews ------------------------------------------------------

    def get_review(self, key: str) -> ReviewRow | None:
        row = self.conn.execute(
            "SELECT key, repo, pr, head_sha, state, verdict, hold_reason "
            "FROM reviews WHERE key = ?",
            (key,),
        ).fetchone()
        return ReviewRow(**dict(row)) if row else None

    def sha_was_judged(self, repo: str, pr: int, head_sha: str) -> bool:
        """True when this exact commit already got a pass or a recorded hold.

        Gate 4's evidence that a re-request brought no new code.
        """
        row = self.conn.execute(
            "SELECT 1 FROM reviews WHERE repo=? AND pr=? AND head_sha=? "
            "AND state IN ('published','held') LIMIT 1",
            (repo, pr, head_sha),
        ).fetchone()
        return row is not None

    def start_review(
        self, *, key: str, repo: str, pr: int, head_sha: str, requested_at: str
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO reviews "
            "(key, repo, pr, head_sha, requested_at, state, created_at) "
            "VALUES (?,?,?,?,?, 'running', ?)",
            (key, repo, pr, head_sha, requested_at, now_ms()),
        )

    def finish_review(
        self,
        key: str,
        *,
        state: str,
        verdict: str | None = None,
        hold_reason: str | None = None,
        cost_usd: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        duration_s: float | None = None,
        transcript: str | None = None,
        model: str | None = None,
        findings: int | None = None,
        blocking: int | None = None,
        should_fix: int | None = None,
        inline: int | None = None,
        summary_findings: int | None = None,
        ci_state: str | None = None,
    ) -> None:
        """Writes only the columns it was given: a full-column UPDATE would null
        whatever the caller left out, and the publish path calls this twice."""
        given = {
            name: value
            for name, value in (
                ("verdict", verdict), ("hold_reason", hold_reason),
                ("cost_usd", cost_usd), ("tokens_in", tokens_in),
                ("tokens_out", tokens_out), ("duration_s", duration_s),
                ("transcript", transcript), ("model", model), ("findings", findings),
                ("blocking", blocking), ("should_fix", should_fix), ("inline", inline),
                ("summary_findings", summary_findings), ("ci_state", ci_state),
            )
            if value is not None
        }
        given["state"] = state
        given["finished_at"] = now_ms()
        # column names are literals from the tuple above, never caller input
        sets = ", ".join(f"{name}=?" for name in given)
        self.conn.execute(
            f"UPDATE reviews SET {sets} WHERE key=?", (*given.values(), key)
        )

    def record_hold(self, *, key: str, repo: str, pr: int, head_sha: str,
                    requested_at: str, reason: str) -> None:
        """A hold we are done with for this key — it will not be retried."""
        self.conn.execute(
            "INSERT OR REPLACE INTO reviews "
            "(key, repo, pr, head_sha, requested_at, state, hold_reason, created_at, finished_at) "
            "VALUES (?,?,?,?,?, 'held', ?, ?, ?)",
            (key, repo, pr, head_sha, requested_at, reason, now_ms(), now_ms()),
        )

    def expire_ci_watch(self, before_ms: int) -> int:
        """Settle approvals whose watch window passed while still waiting —
        without this they freeze on the panel forever, unsweepable and
        unsettled (a daemon that slept through the window leaves them so)."""
        cur = self.conn.execute(
            "UPDATE reviews SET ci_state='expired' "
            "WHERE verdict='ok' AND state='published' AND ci_state='waiting' "
            "AND created_at < ?",
            (before_ms,),
        )
        self.conn.commit()
        return cur.rowcount

    def watching_ci(self, since_ms: int) -> list[sqlite3.Row]:
        """Approvals still waiting on the build robbie asked for."""
        return list(
            self.conn.execute(
                "SELECT key, repo, pr, head_sha, model, created_at FROM reviews "
                "WHERE verdict='ok' AND state='published' AND ci_state='waiting' "
                "AND created_at >= ? ORDER BY created_at",
                (since_ms,),
            )
        )

    def set_ci_state(self, key: str, state: str) -> None:
        self.conn.execute(
            "UPDATE reviews SET ci_state=?, ci_seen_at=? WHERE key=?", (state, now_ms(), key)
        )

    def approved_and_green(self, since_ms: int) -> list[sqlite3.Row]:
        """What a human could pick up: robbie approved it, the build went green, and
        somebody is still asking for the review.

        The newest pass per PR and no other. A push while a review request is open
        keeps `requested_at` and only moves `head_sha`, so a busy PR collects one
        row per commit — listing them all repeats the PR once per commit that no
        longer exists, and an `ok` two commits back keeps claiming the PR is
        approved after a later pass said needs-work. Rank first, judge after: a PR
        whose latest pass is not an `ok` has to fall out, which it cannot do if the
        verdict is part of what picks the row.

        The `requested` join is the other half of "ready for a human". A
        changes-requested review consumes the request, and an author who never asks
        again leaves an approval nobody is waiting on — true, and not a board row.
        Until the first tick of a repo fills the table, that repo shows nothing.
        """
        return list(
            self.conn.execute(
                "SELECT repo, pr, head_sha, model, verdict, ci_state, ci_seen_at, created_at "
                "FROM (SELECT *, ROW_NUMBER() OVER "
                # id breaks the tie: two passes can land in the same millisecond
                "             (PARTITION BY repo, pr ORDER BY created_at DESC, id DESC) newest "
                "      FROM reviews WHERE state='published' AND created_at >= ?) r "
                "WHERE newest = 1 AND verdict='ok' AND ci_state IN ('green','waiting','red') "
                "  AND EXISTS (SELECT 1 FROM requested q "
                "              WHERE q.repo = r.repo AND q.pr = r.pr) "
                "ORDER BY ci_state='green' DESC, created_at DESC",
                (since_ms,),
            )
        )

    def set_requested(self, repo: str, prs: Collection[int]) -> None:
        """Replace what this repo's queue read saw asking for our review.

        One transaction: the panel reads this concurrently, and a reader landing
        between the delete and the inserts would show an empty board.
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("DELETE FROM requested WHERE repo=?", (repo,))
            self.conn.executemany(
                "INSERT INTO requested (repo, pr, seen_at) VALUES (?,?,?)",
                [(repo, int(pr), now_ms()) for pr in prs],
            )
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def approved_prs(self, repo: str) -> list[int]:
        """PRs whose newest judged pass is an `ok`, whatever the build said after.

        Same rank-first-judge-after shape as the panel, and deliberately without
        its `ci_state` and `requested` conditions: this answers "did we end up
        approving this", which is what decides whether a standing rejection of ours
        still means anything. No window — the caller's PR set is already bounded by
        what is open under the label.
        """
        return [
            int(r["pr"]) for r in self.conn.execute(
                "SELECT pr FROM (SELECT pr, verdict, ROW_NUMBER() OVER "
                "             (PARTITION BY pr ORDER BY created_at DESC, id DESC) newest "
                "      FROM reviews WHERE repo=? AND state='published') "
                "WHERE newest = 1 AND verdict='ok' ORDER BY pr",
                (repo,),
            )
        ]

    def unrequest(self, repo: str, prs: Collection[int]) -> int:
        """Take these PRs off the board without settling anything.

        Deliberately not a `ci_state`: a brake label comes off as easily as it goes
        on, and a settled row never comes back — it would need a fresh pass to
        return, which gate 4 refuses when the commit has not moved. Dropping the
        row here instead means the next tick's queue read puts the PR back on the
        board the moment the label is gone.
        """
        holes = ",".join("?" * len(prs))
        if not prs:
            return 0
        cur = self.conn.execute(
            f"DELETE FROM requested WHERE repo=? AND pr IN ({holes})",
            (repo, *(int(pr) for pr in prs)),
        )
        return cur.rowcount or 0

    def settle_stale(self, repo: str, heads: dict[int, str]) -> int:
        """Retire approvals whose commit is no longer the PR's head.

        `ci_watch` only revisits rows still waiting on a build, so an approval that
        already went green is never looked at again — a push after it leaves the
        board claiming a commit that no longer exists. This is the sweep that
        notices, from a read the tick was making anyway.
        """
        settled = 0
        for pr, head in heads.items():
            cur = self.conn.execute(
                "UPDATE reviews SET ci_state='stale' WHERE repo=? AND pr=? AND head_sha != ? "
                "AND ci_state IN ('green','waiting','red')",
                (repo, int(pr), head),
            )
            settled += cur.rowcount or 0
        return settled

    def settle_done(self, repo: str, pr: int) -> int:
        """A human has taken this PR; retire it from the panel and the sweep.

        Keyed on (repo, pr): building a review key needs the timeline read this gate
        exists to skip. Every row, not just the approval.
        """
        cur = self.conn.execute(
            "UPDATE reviews SET ci_state='done' WHERE repo=? AND pr=? "
            "AND COALESCE(ci_state,'') != 'done'",
            (repo, pr),
        )
        return cur.rowcount or 0

    def settle_unlabeled(self, repo: str, labeled: Collection[int]) -> int:
        """Retire from the panel every PR of this repo outside `labeled`.

        Merged, closed, or the label taken off — all three read the same way here,
        and none of them can reach `settle_done`: that gate only sees PRs the queue
        still returns. Unlike 'done' this leaves the reply sweep alone. An author
        answering findings on a PR that lost the label still deserves an answer;
        it just is not something a human can pick up any more.
        """
        holes = ",".join("?" * len(labeled))
        # an empty list is a repo with nothing under the label, not a no-op
        keep = f" AND pr NOT IN ({holes})" if labeled else ""
        cur = self.conn.execute(
            f"UPDATE reviews SET ci_state='unlabeled' WHERE repo=?{keep} "
            "AND COALESCE(ci_state,'') NOT IN ('done','unlabeled')",
            (repo, *labeled),
        )
        return cur.rowcount or 0

    def passes_for(self, repo: str, pr: int) -> list[sqlite3.Row]:
        """Earlier judged passes on this PR, oldest first."""
        return list(
            self.conn.execute(
                "SELECT head_sha, verdict, state, hold_reason, created_at FROM reviews "
                "WHERE repo=? AND pr=? AND state IN ('published','held') "
                "ORDER BY created_at",
                (repo, pr),
            )
        )

    def reviewed_prs(self, repo: str, *, since_ms: int = 0) -> list[int]:
        """PRs reviewed since `since_ms` — where threads of ours can exist.

        Windowed because each costs an API read every tick and the list only grows.
        A PR a human has taken drops out early for the same reason. Threads only
        exist where a pass anchored comments to the diff, so a pass that anchored
        nothing is excluded too: reading it every tick buys nothing, and those reads
        are what drain the user-wide GraphQL pool.

        `inline` is three-valued, and only the recorded 0 means "this pass opened no
        threads". NULL means no pass ever wrote the column — rows older than it, and
        every approval, since an ok returns before the publish path that records it.
        Those PRs keep their place in the sweep: the threads it answers are every
        thread the reviewer login opened, which is not a set this table can bound.
        """
        return [
            int(r["pr"]) for r in self.conn.execute(
                "SELECT DISTINCT pr FROM reviews r WHERE repo=? AND state='published' "
                "AND created_at >= ? AND COALESCE(ci_state,'') != 'done' "
                "AND EXISTS (SELECT 1 FROM reviews c WHERE c.repo=r.repo AND c.pr=r.pr "
                "AND c.state='published' AND COALESCE(c.inline,1) > 0) "
                "ORDER BY pr DESC",
                (repo, since_ms),
            )
        ]

    # ----- what the panel reads ------------------------------------------
    #
    # Here rather than in dashboard.py so a column added above is one place to
    # check, not two.

    def counts(self) -> tuple[int, int, int]:
        """(reviewing right now, published, rows in total)."""
        running = self.conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE state='running'"
        ).fetchone()[0]
        total, published = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(state='published'), 0) FROM reviews"
        ).fetchone()
        return int(running), int(published), int(total)

    def by_model(self) -> dict[str, sqlite3.Row]:
        """Published runs per model — the arm comparison, keyed by model tag."""
        return {
            r["model"]: r
            for r in self.conn.execute(
                "SELECT model, COUNT(*) runs, SUM(COALESCE(summary_findings, 0)) f, "
                "CAST(AVG(duration_s) AS INT) secs FROM reviews "
                "WHERE state='published' AND model IS NOT NULL GROUP BY model"
            )
        }

    def unfinished(self, limit: int) -> list[sqlite3.Row]:
        """Reviews that are running, held or failed, newest first."""
        return list(
            self.conn.execute(
                "SELECT repo, pr, state, hold_reason, created_at FROM reviews "
                "WHERE state IN ('running','held','failed') ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        )

    def recent_published(self, limit: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT repo, pr, head_sha, verdict, model, findings, summary_findings, "
                "inline, duration_s, tokens_out, created_at, finished_at FROM reviews "
                "WHERE state='published' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        )

    def reap_running(self) -> int:
        """Mark orphaned 'running' rows failed at boot: a row is only 'running'
        while a process holds the container, and that process is gone."""
        cur = self.conn.execute(
            "UPDATE reviews SET state='failed', hold_reason='orchestrator restarted', "
            "finished_at=? WHERE state='running'",
            (now_ms(),),
        )
        return cur.rowcount or 0

    def record_spend(
        self, *, repo: str, pr: int, kind: str,
        cost_usd: float | None, duration_s: float | None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO spend (repo, pr, kind, cost_usd, duration_s, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (repo, pr, kind, cost_usd, duration_s, now_ms()),
        )

    def verdicts_since(self, since_ms: int) -> dict[str, int]:
        """Published verdicts since `since_ms`, by verdict — the heartbeat's tally."""
        return {
            str(r["verdict"]): int(r["n"]) for r in self.conn.execute(
                "SELECT verdict, COUNT(*) AS n FROM reviews WHERE state='published' "
                "AND finished_at >= ? GROUP BY verdict",
                (since_ms,),
            )
        }

    def failed_runs_since(self, since_ms: int) -> int:
        """Failed containers since `since_ms` — from append-only spend, because a
        retry's INSERT OR REPLACE erases the failed review row it recovers."""
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM spend WHERE kind='failed' AND created_at >= ?",
            (since_ms,),
        ).fetchone()[0])

    def spend_since(self, since_ms: int, exclude_models: tuple[str, ...] = ()) -> float:
        """Dollars the account was billed since `since_ms`.

        A third-party endpoint's `cost_usd` is the CLI's own price table, never the
        account's bill, so counting it trips `daily_usd` on money nobody spent.
        """
        holes = ",".join("?" * len(exclude_models))
        skip = f" AND COALESCE(model, '') NOT IN ({holes})" if exclude_models else ""
        row = self.conn.execute(
            "SELECT COALESCE((SELECT SUM(cost_usd) FROM reviews "
            f"                WHERE created_at >= ?{skip}), 0.0) "
            "     + COALESCE((SELECT SUM(cost_usd) FROM spend WHERE created_at >= ?), 0.0) "
            "AS usd",
            (since_ms, *exclude_models, since_ms),
        ).fetchone()
        return float(row["usd"])

    # ----- meters ---------------------------------------------------------

    def read_meter(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM meters WHERE name=?", (name,)).fetchone()

    def write_meter(self, name: str, *, pct: float, note: str) -> None:
        self.conn.execute(
            "INSERT INTO meters (name, pct, note, read_at) VALUES (?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET pct=excluded.pct, note=excluded.note, "
            "read_at=excluded.read_at, error=NULL, error_at=NULL",
            (name, pct, note, now_ms()),
        )

    def meter_failed(self, name: str, error: str) -> None:
        """A failed read is not a reading: `pct` and `read_at` survive it, and how
        old they are is what decides whether the gate may still use them."""
        self.conn.execute(
            "INSERT INTO meters (name, error, error_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET error=excluded.error, error_at=excluded.error_at",
            (name, error, now_ms()),
        )

    # ----- notices / seeding --------------------------------------------

    def notice_seen(self, key: str) -> bool:
        """Whether this key was ever recorded, without recording it."""
        return (
            self.conn.execute("SELECT 1 FROM notices WHERE key=?", (key,)).fetchone()
            is not None
        )

    def notice_once(self, key: str) -> bool:
        """True the first time this key is seen; False every time after."""
        try:
            self.conn.execute(
                "INSERT INTO notices (key, created_at) VALUES (?, ?)", (key, now_ms())
            )
        except sqlite3.IntegrityError:
            return False
        return True

    def is_seeded(self, repo: str) -> bool:
        return (
            self.conn.execute("SELECT 1 FROM seeded WHERE repo=?", (repo,)).fetchone()
            is not None
        )

    def mark_seeded(self, repo: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO seeded (repo, seeded_at) VALUES (?, ?)", (repo, now_ms())
        )
