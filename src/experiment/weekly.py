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
    "Facts",
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
    "apply",
    "apply_error",
    "decide",
    "tick",
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


@dataclasses.dataclass(frozen=True)
class Facts:
    """What a tick observed this time, for :func:`decide` to act on.

    Bundled rather than passed as separate arguments since every phase of
    ``decide`` needs a different subset, and a new fact (e.g. a second
    remote to compare against) should not change every phase's signature.

    Attributes:
        now: The current time.
        remote_sha: The commit that should be benchmarked next.
        job_status: The dispatched job's status, if one is in flight and
            its token wasn't lost to a crash; ``None`` otherwise.
        history: The suite's durable completion history.
    """

    now: datetime
    remote_sha: str
    job_status: JobStatus | None
    history: DurableHistory


def _decide_waiting(
    state: TransientState, facts: Facts, config: WeeklyBenchmarkConfig
) -> tuple[TransientState, Action]:
    """``decide``'s ``WAITING`` branch: wait, or dispatch what's due."""
    if facts.now < state.next_wake_at:
        return state, Sleep()

    if facts.remote_sha == facts.history.last_completed_sha:
        next_wake_at = config.next_scheduled_wake(facts.now)
        return dataclasses.replace(state, next_wake_at=next_wake_at), Sleep()

    dispatching = dataclasses.replace(
        state,
        phase=Phase.DISPATCHED,
        dispatch_sha=facts.remote_sha,
        dispatch_token=None,
        attempt_count=0,
        last_error=None,
    )
    return dispatching, Submit(facts.remote_sha)


def _decide_dispatched(
    state: TransientState, facts: Facts, config: WeeklyBenchmarkConfig
) -> tuple[TransientState, Action]:
    """``decide``'s ``DISPATCHED`` branch: poll, resubmit, or finish."""
    if state.dispatch_sha is None:
        raise ValueError("DISPATCHED state has no dispatch_sha")

    if state.dispatch_token is None or facts.job_status is None:
        # The token was never persisted (a crash between Submit's side
        # effect and apply()) or this tick didn't poll - resubmit.
        # Dispatcher.submit's own dedup makes this safe against a job
        # that's already running or already finished.
        return state, Submit(state.dispatch_sha)

    if facts.job_status is JobStatus.RUNNING:
        next_wake_at = config.poll_interval(facts.now)
        return dataclasses.replace(state, next_wake_at=next_wake_at), Sleep()

    if facts.job_status is JobStatus.SUCCEEDED:
        finishing = dataclasses.replace(state, phase=Phase.FINISHING)
        return finishing, Finish(state.dispatch_sha, state.dispatch_token)

    return state, MarkFailed(f"job for {state.dispatch_sha} failed")


def _decide_finishing(state: TransientState) -> tuple[TransientState, Action]:
    """``decide``'s ``FINISHING`` branch: retry until publish succeeds.

    Always re-issues ``Finish`` - safe because ``Publisher.publish`` is
    required to be idempotent, so a crash between a prior Finish attempt
    and ``apply`` just means retrying, not duplicating anything.
    """
    if state.dispatch_sha is None or state.dispatch_token is None:
        raise ValueError("FINISHING state is missing dispatch_sha/dispatch_token")
    return state, Finish(state.dispatch_sha, state.dispatch_token)


def _decide_failed(
    state: TransientState, facts: Facts, config: WeeklyBenchmarkConfig
) -> tuple[TransientState, Action]:
    """``decide``'s ``FAILED`` branch: retry a fixable failure, or hold.

    Behaves like ``WAITING`` for due-checking, so a transient failure (a
    cluster outage, say) is retried on the same cadence - but
    ``attempt_count`` carries across ``FAILED`` and once it reaches
    ``config.max_attempts`` for this sha, holds rather than retrying every
    tick, until a new commit gives it a fresh attempt budget.
    """
    if state.dispatch_sha is None:
        raise ValueError("FAILED state has no dispatch_sha")

    if facts.remote_sha != state.dispatch_sha:
        reset = dataclasses.replace(
            state,
            phase=Phase.WAITING,
            next_wake_at=facts.now,
            attempt_count=0,
            last_error=None,
        )
        return _decide_waiting(reset, facts, config)

    if state.attempt_count >= config.max_attempts or facts.now < state.next_wake_at:
        return state, Sleep()

    dispatching = dataclasses.replace(
        state, phase=Phase.DISPATCHED, dispatch_token=None
    )
    return dispatching, Submit(state.dispatch_sha)


def decide(
    state: TransientState, facts: Facts, config: WeeklyBenchmarkConfig
) -> tuple[TransientState, Action]:
    """Pure transition function - no I/O, no side effects.

    Returns the state to persist *before* the returned ``Action``'s side
    effect runs, paired with that ``Action``. Never invents a dispatch
    token, completion timestamp, or publish id - those only exist after
    their side effect has actually run; ``apply``/``apply_error`` fold
    such a result back into state once the caller has it.
    """
    if state.phase is Phase.WAITING:
        return _decide_waiting(state, facts, config)
    if state.phase is Phase.DISPATCHED:
        return _decide_dispatched(state, facts, config)
    if state.phase is Phase.FINISHING:
        return _decide_finishing(state)
    return _decide_failed(state, facts, config)


def apply(
    state: TransientState,
    action: Action,
    result: str | None,
    now: datetime,
    config: WeeklyBenchmarkConfig,
) -> tuple[TransientState, DurableHistory | None]:
    """Fold a *successful* action's result into state.

    ``result`` is the ``Dispatcher.submit`` token for ``Submit``, the
    ``Publisher.publish`` identifier for ``Finish``, and unused for
    ``Sleep``/``MarkFailed``. Returns the updated ``TransientState``
    always, and a ``DurableHistory`` update only for ``Finish`` - the only
    transition that advances ``last_completed_sha``/``last_completed_at``.
    """
    if isinstance(action, Sleep):
        return state, None

    if isinstance(action, Submit):
        if result is None:
            raise ValueError("Submit requires a dispatch token")
        return dataclasses.replace(state, dispatch_token=result), None

    if isinstance(action, Finish):
        if result is None:
            raise ValueError("Finish requires a publish id")
        completed = dataclasses.replace(
            state,
            phase=Phase.WAITING,
            next_wake_at=config.next_scheduled_wake(now),
            dispatch_sha=None,
            dispatch_token=None,
            last_publish_id=result,
            attempt_count=0,
            last_error=None,
        )
        history = DurableHistory(last_completed_sha=action.sha, last_completed_at=now)
        return completed, history

    failed = dataclasses.replace(
        state,
        phase=Phase.FAILED,
        attempt_count=state.attempt_count + 1,
        last_error=action.reason,
    )
    return failed, None


def apply_error(state: TransientState, action: Action, error: str) -> TransientState:
    """Fold a *failed* action's side effect into state.

    Settles in ``Phase.FAILED`` with ``attempt_count`` incremented and
    ``last_error`` recorded, regardless of whether ``action`` was
    ``Submit`` or ``Finish``: this is what closes the "action raised,
    pre-action state left on disk forever" gap - the next tick sees
    ``FAILED`` and ``decide`` (via ``_decide_failed``) determines whether
    to retry, rather than the same exception recurring against a state
    that never changed.
    """
    if isinstance(action, Sleep | MarkFailed):
        raise ValueError(f"{action} has no side effect that can fail")
    return dataclasses.replace(
        state,
        phase=Phase.FAILED,
        attempt_count=state.attempt_count + 1,
        last_error=error,
    )


def _perform(action: Action, config: WeeklyBenchmarkConfig) -> str | None:
    """Run an action's side effect, returning what `apply` needs to fold in."""
    if isinstance(action, Submit):
        return config.dispatcher.submit(action.sha)
    if isinstance(action, Finish):
        artifacts = config.reporter(action.sha, config.experiment, config.out_dir)
        return config.publisher.publish(action.sha, artifacts)
    return None


def tick(config: WeeklyBenchmarkConfig, now: datetime) -> TransientState:
    """Single stateless invocation: load, decide, act, persist.

    Acquires ``config.lock`` first and does nothing if another tick holds
    it - what makes a cheap, frequent external trigger cadence safe
    against a slow-running tick overlapping the next one. Persists the
    pre-action state *before* performing the action's side effect, so a
    crash mid-action leaves a recoverable state rather than one that
    looks like nothing happened.
    """
    held = config.lock.try_acquire()
    if held is None:
        return config.transient_store.load() or TransientState.initial(now)

    with held:
        state = config.transient_store.load() or TransientState.initial(now)
        idle = state.phase in (Phase.WAITING, Phase.FAILED)
        if idle and now < state.next_wake_at:
            return state

        job_status = None
        if state.phase is Phase.DISPATCHED and state.dispatch_token is not None:
            job_status = config.dispatcher.poll(state.dispatch_token)
        history = config.history_store.load() or DurableHistory(None, None)
        observed = Facts(
            now=now,
            remote_sha=config.remote_sha(),
            job_status=job_status,
            history=history,
        )

        pre_action_state, action = decide(state, observed, config)
        config.transient_store.save(pre_action_state)
        if isinstance(action, Sleep):
            return pre_action_state

        try:
            result = _perform(action, config)
        except Exception as exc:
            failed_state = apply_error(pre_action_state, action, str(exc))
            config.transient_store.save(failed_state)
            return failed_state

        final_state, history_update = apply(
            pre_action_state, action, result, now, config
        )
        config.transient_store.save(final_state)
        if history_update is not None:
            config.history_store.save(history_update)
        return final_state
