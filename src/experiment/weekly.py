"""Periodic benchmark scheduling: a stateless, crash-safe orchestrator.

A ``tick`` is invoked on some external periodic cadence (a cron entry, a
systemd timer) and does one unit of work before exiting: check whether a
benchmark run is due, dispatch it, poll it, or publish its results. All
state needed between invocations is externalized - nothing here assumes
consecutive ticks run in the same process, or even on the same host.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from experiment.design import Experiment

__all__ = [
    "Dispatcher",
    "DurableHistory",
    "JobStatus",
    "Phase",
    "Publisher",
    "Reporter",
    "TransientState",
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
