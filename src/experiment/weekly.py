"""Periodic benchmark scheduling: a stateless, crash-safe orchestrator.

A ``tick`` is invoked on some external periodic cadence (a cron entry, a
systemd timer) and does one unit of work before exiting: check whether a
benchmark run is due, dispatch it, poll it, or publish its results. All
state needed between invocations is externalized - nothing here assumes
consecutive ticks run in the same process, or even on the same host.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from experiment.design import Experiment

__all__ = [
    "Action",
    "Dispatcher",
    "DurableHistory",
    "Finish",
    "HistoryStore",
    "JobStatus",
    "Lock",
    "MarkFailed",
    "Phase",
    "Publisher",
    "Reporter",
    "Sleep",
    "Submit",
    "TransientState",
    "TransientStore",
    "WeeklyBenchmarkConfig",
]


class Phase(StrEnum):
    """Where a benchmark suite is in its schedule."""

    WAITING = "waiting"
    DISPATCHED = "dispatched"
    FINISHING = "finishing"
    FAILED = "failed"


class JobStatus(StrEnum):
    """What a dispatched job's compute backend reports."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class TransientState:
    """Runtime scheduling state, rewritten every tick.

    Deliberately kept out of git-tracked storage: a concrete
    :class:`TransientStore` writing this to a tracked file would dirty the
    working tree right before a dispatch that requires a clean tree.

    Attributes:
        phase: Where the suite currently is in its schedule.
        next_wake_at: When a ``WAITING`` or ``FAILED`` phase should next
            reconsider whether a run is due.
        dispatch_sha: The commit currently being worked on, or ``None``.
        dispatch_token: The compute backend's handle for the in-flight job,
            or ``None`` if none has been (successfully) submitted yet.
        last_publish_id: The last successful publish's identifier (e.g. a
            PR URL), or ``None``.
        attempt_count: Consecutive failures for ``dispatch_sha``.
        last_error: The most recent failure's message, or ``None``.
    """

    phase: Phase
    next_wake_at: datetime
    dispatch_sha: str | None
    dispatch_token: str | None
    last_publish_id: str | None
    attempt_count: int
    last_error: str | None

    @staticmethod
    def initial(now: datetime) -> TransientState:
        """The state a suite with no prior history starts from.

        Args:
            now: The current time, used as an immediately-due wake time.

        Returns:
            A ``WAITING`` state with no dispatch history.
        """
        return TransientState(
            phase=Phase.WAITING,
            next_wake_at=now,
            dispatch_sha=None,
            dispatch_token=None,
            last_publish_id=None,
            attempt_count=0,
            last_error=None,
        )


@dataclasses.dataclass(frozen=True)
class DurableHistory:
    """Which commit was last successfully benchmarked, and when.

    Safe to persist to git-tracked storage: only ever written at a
    successful finish, never mid-dispatch.

    Attributes:
        last_completed_sha: The commit last successfully benchmarked, or
            ``None`` if none has completed yet.
        last_completed_at: When that run completed, or ``None``.
    """

    last_completed_sha: str | None
    last_completed_at: datetime | None


class Dispatcher(Protocol):
    """Submits and polls a benchmark run on some compute backend."""

    def submit(self, sha: str) -> str:
        """Start work for ``sha``; return an opaque token for the job.

        Must derive the job's identity deterministically from ``sha`` (e.g.
        a job name) and check for an already-running or already-completed
        job with that identity before submitting: a crash between a prior
        submission and its token being persisted means this can be called
        again for a ``sha`` that already has work in flight or finished,
        and must not result in duplicate work.

        Is also responsible for making the dispatched work actually
        reflect ``sha`` (e.g. checking out that commit before packaging) -
        the scheduler only ever hands over a sha, never a working tree.

        Args:
            sha: The commit to benchmark.

        Returns:
            An opaque token identifying the submitted job.
        """

    def poll(self, token: str) -> JobStatus:
        """Check a previously submitted job's status.

        Args:
            token: A token previously returned by :meth:`submit`.

        Returns:
            The job's current status.
        """


class Reporter(Protocol):
    """Renders a completed run's results into shippable artifacts."""

    def __call__(
        self, sha: str, experiment: Experiment, out_dir: Path
    ) -> Sequence[Path]:
        """Render ``sha``'s stored results into artifacts under ``out_dir``.

        Args:
            sha: The commit whose results to render.
            experiment: The experiment the results were stored under.
            out_dir: Where to write rendered artifacts.

        Returns:
            The paths written under ``out_dir``.
        """


class Publisher(Protocol):
    """Publishes rendered artifacts somewhere a human will see them."""

    def publish(self, sha: str, artifacts: Sequence[Path]) -> str:
        """Publish ``artifacts`` as the results for ``sha``.

        Must be idempotent: calling this again for a ``sha`` that already
        has a publication returns the existing identifier rather than
        erroring or duplicating it. This can't be delegated to the
        underlying tool - e.g. ``gh pr create`` fails outright for a branch
        that already has an open PR.

        Args:
            sha: The commit the artifacts are for.
            artifacts: Paths previously returned by a :class:`Reporter`.

        Returns:
            An identifier for the publication (e.g. a PR URL).
        """


class TransientStore(Protocol):
    """Reads and writes a suite's runtime scheduling state."""

    def load(self) -> TransientState | None:
        """Return the stored state, or ``None`` if none has been saved yet."""

    def save(self, state: TransientState) -> None:
        """Persist ``state`` atomically.

        A partial write (e.g. a crash mid-write) must never leave a file
        that fails to parse on the next :meth:`load`. Must write to a path
        outside of git tracking.
        """


class HistoryStore(Protocol):
    """Reads and writes a suite's durable completion history."""

    def load(self) -> DurableHistory | None:
        """Return the stored history, or ``None`` if none exists yet."""

    def save(self, history: DurableHistory) -> None:
        """Persist ``history``.

        Safe for a concrete implementation to commit to git: this is only
        ever called after a successful finish, never mid-dispatch.
        """


class Lock(Protocol):
    """Coordinates against overlapping ticks."""

    def try_acquire(self) -> AbstractContextManager[None] | None:
        """Attempt to acquire, without blocking.

        Returns:
            An already-acquired context manager on success, whose
            ``__enter__`` is a no-op and ``__exit__`` releases it; or
            ``None`` if another tick currently holds it, in which case the
            caller must do no work this tick.
        """


@dataclasses.dataclass(frozen=True)
class Sleep:
    """Nothing to do; only ``next_wake_at`` needs persisting."""


@dataclasses.dataclass(frozen=True)
class Submit:
    """Submit ``sha`` to the :class:`Dispatcher`."""

    sha: str


@dataclasses.dataclass(frozen=True)
class Finish:
    """Render and publish ``sha``'s results."""

    sha: str
    dispatch_token: str


@dataclasses.dataclass(frozen=True)
class MarkFailed:
    """Settle in :attr:`Phase.FAILED` with ``reason`` recorded."""

    reason: str


Action = Sleep | Submit | Finish | MarkFailed


@dataclasses.dataclass(frozen=True)
class WeeklyBenchmarkConfig:
    """Wires a generic scheduler up to one concrete deployment.

    Attributes:
        label: Identifies this suite, for job names and file paths.
        experiment: The experiment being benchmarked.
        remote_sha: Returns the commit that should be benchmarked next,
            e.g. the tip of a tracked branch.
        next_scheduled_wake: Given a completion time, returns the next
            calendar-anchored wake time (e.g. "next Sunday 00:00 UTC").
            A fixed calendar anchor is used rather than a duration so the
            schedule doesn't drift later every cycle a run overruns.
        poll_interval: Given now, returns when to next check a dispatched
            job.
        dispatcher: Submits and polls benchmark runs.
        reporter: Renders a completed run's results.
        publisher: Publishes rendered results.
        transient_store: Reads and writes runtime scheduling state.
        history_store: Reads and writes durable completion history.
        lock: Coordinates against overlapping ticks.
        out_dir: Where the reporter should write rendered artifacts.
        max_attempts: Consecutive failures for one sha before giving up
            and settling in ``FAILED`` instead of retrying every tick.
    """

    label: str
    experiment: Experiment
    remote_sha: Callable[[], str]
    next_scheduled_wake: Callable[[datetime], datetime]
    poll_interval: Callable[[datetime], datetime]
    dispatcher: Dispatcher
    reporter: Reporter
    publisher: Publisher
    transient_store: TransientStore
    history_store: HistoryStore
    lock: Lock
    out_dir: Path
    max_attempts: int = 3
