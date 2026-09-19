"""Tests for the weekly benchmark scheduler (``experiment.weekly``).

Exercises ``decide`` in isolation - a pure function - so these need no
real Dispatcher/Reporter/Publisher/stores; a config carrying fakes that
raise on use is enough to prove decide() never touches them.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import pytest

from experiment.design import Component, Experiment
from experiment.weekly import (
    DurableHistory,
    Facts,
    Finish,
    JobStatus,
    MarkFailed,
    Phase,
    Sleep,
    Submit,
    TransientState,
    WeeklyBenchmarkConfig,
    apply,
    apply_error,
    decide,
)

NOW = datetime(2026, 1, 1, 12, 0)


def _unused(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("decide() must not touch injected collaborators")


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    return Experiment(
        name="bench",
        components=[Component(name="a", config=object(), seeds=[0])],
        results_dir=tmp_path,
    )


@pytest.fixture
def config(experiment) -> WeeklyBenchmarkConfig:
    return WeeklyBenchmarkConfig(
        label="bench",
        experiment=experiment,
        remote_sha=_unused,
        next_scheduled_wake=lambda now: now + timedelta(days=7),
        poll_interval=lambda now: now + timedelta(minutes=20),
        dispatcher=_unused,  # type: ignore[arg-type]
        reporter=_unused,  # type: ignore[arg-type]
        publisher=_unused,  # type: ignore[arg-type]
        transient_store=_unused,  # type: ignore[arg-type]
        history_store=_unused,  # type: ignore[arg-type]
        lock=_unused,  # type: ignore[arg-type]
        out_dir=None,  # type: ignore[arg-type]
        max_attempts=3,
    )


def waiting(**overrides) -> TransientState:
    base = TransientState(
        phase=Phase.WAITING,
        next_wake_at=NOW,
        dispatch_sha=None,
        dispatch_token=None,
        last_publish_id=None,
        attempt_count=0,
        last_error=None,
    )
    return dataclasses.replace(base, **overrides)


def facts(**overrides) -> Facts:
    base = Facts(
        now=NOW,
        remote_sha="abc123",
        job_status=None,
        history=DurableHistory(last_completed_sha=None, last_completed_at=None),
    )
    return dataclasses.replace(base, **overrides)


# --- WAITING ------------------------------------------------------------


def test_waiting_before_next_wake_sleeps_untouched(config):
    state = waiting(next_wake_at=NOW + timedelta(days=1))
    new_state, action = decide(state, facts(), config)
    assert (new_state, action) == (state, Sleep())


def test_waiting_due_with_nothing_new_advances_the_wake_time(config):
    state = waiting(next_wake_at=NOW)
    history = DurableHistory(last_completed_sha="abc123", last_completed_at=NOW)
    observed = facts(remote_sha="abc123", history=history)
    new_state, action = decide(state, observed, config)
    assert action == Sleep()
    assert new_state.next_wake_at == NOW + timedelta(days=7)


def test_waiting_due_with_a_new_sha_dispatches_it(config):
    state = waiting(next_wake_at=NOW)
    new_state, action = decide(state, facts(remote_sha="new-sha"), config)
    assert action == Submit("new-sha")
    assert new_state.phase == Phase.DISPATCHED
    assert new_state.dispatch_sha == "new-sha"
    assert new_state.dispatch_token is None


# --- DISPATCHED -----------------------------------------------------------


def test_dispatched_with_no_token_resubmits(config):
    state = waiting(
        phase=Phase.DISPATCHED, dispatch_sha="s1", dispatch_token=None
    )
    new_state, action = decide(state, facts(job_status=JobStatus.RUNNING), config)
    assert (new_state, action) == (state, Submit("s1"))


def test_dispatched_unpolled_resubmits(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1", dispatch_token="t1")
    new_state, action = decide(state, facts(job_status=None), config)
    assert (new_state, action) == (state, Submit("s1"))


def test_dispatched_running_sleeps_and_reschedules_the_poll(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1", dispatch_token="t1")
    new_state, action = decide(state, facts(job_status=JobStatus.RUNNING), config)
    assert action == Sleep()
    assert new_state.next_wake_at == NOW + timedelta(minutes=20)


def test_dispatched_succeeded_moves_to_finishing(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1", dispatch_token="t1")
    new_state, action = decide(state, facts(job_status=JobStatus.SUCCEEDED), config)
    assert action == Finish("s1", "t1")
    assert new_state.phase == Phase.FINISHING


def test_dispatched_failed_marks_it_failed(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1", dispatch_token="t1")
    new_state, action = decide(state, facts(job_status=JobStatus.FAILED), config)
    assert action == MarkFailed("job for s1 failed")
    assert new_state == state


# --- FINISHING --------------------------------------------------------------


def test_finishing_always_retries_the_finish(config):
    state = waiting(phase=Phase.FINISHING, dispatch_sha="s1", dispatch_token="t1")
    new_state, action = decide(state, facts(), config)
    assert (new_state, action) == (state, Finish("s1", "t1"))


# --- FAILED -----------------------------------------------------------------


def test_failed_with_a_new_sha_dispatches_it_immediately(config):
    state = waiting(
        phase=Phase.FAILED,
        dispatch_sha="s1",
        next_wake_at=NOW + timedelta(days=3),
        attempt_count=3,
        last_error="boom",
    )
    new_state, action = decide(state, facts(remote_sha="s2"), config)
    assert action == Submit("s2")
    assert new_state.phase == Phase.DISPATCHED
    assert new_state.dispatch_sha == "s2"
    assert new_state.attempt_count == 0
    assert new_state.last_error is None


def test_failed_at_max_attempts_holds(config):
    state = waiting(phase=Phase.FAILED, dispatch_sha="s1", attempt_count=3)
    new_state, action = decide(state, facts(remote_sha="s1"), config)
    assert (new_state, action) == (state, Sleep())


def test_failed_before_next_wake_holds(config):
    state = waiting(
        phase=Phase.FAILED,
        dispatch_sha="s1",
        attempt_count=1,
        next_wake_at=NOW + timedelta(hours=1),
    )
    new_state, action = decide(state, facts(remote_sha="s1"), config)
    assert (new_state, action) == (state, Sleep())


def test_failed_due_and_under_the_limit_retries(config):
    state = waiting(phase=Phase.FAILED, dispatch_sha="s1", attempt_count=1)
    new_state, action = decide(state, facts(remote_sha="s1"), config)
    assert action == Submit("s1")
    assert new_state.phase == Phase.DISPATCHED
    assert new_state.dispatch_token is None


# --- apply --------------------------------------------------------------


def test_apply_sleep_is_a_no_op(config):
    state = waiting()
    new_state, history = apply(state, Sleep(), None, NOW, config)
    assert (new_state, history) == (state, None)


def test_apply_submit_stores_the_token(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1")
    new_state, history = apply(state, Submit("s1"), "tok-1", NOW, config)
    assert new_state.dispatch_token == "tok-1"
    assert history is None


def test_apply_submit_without_a_token_raises(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1")
    with pytest.raises(ValueError, match="dispatch token"):
        apply(state, Submit("s1"), None, NOW, config)


def test_apply_finish_resets_to_waiting_and_records_history(config):
    state = waiting(
        phase=Phase.FINISHING, dispatch_sha="s1", dispatch_token="t1", attempt_count=2
    )
    new_state, history = apply(state, Finish("s1", "t1"), "pr-url", NOW, config)
    assert new_state.phase == Phase.WAITING
    assert new_state.next_wake_at == NOW + timedelta(days=7)
    assert (new_state.dispatch_sha, new_state.dispatch_token) == (None, None)
    assert new_state.last_publish_id == "pr-url"
    assert new_state.attempt_count == 0
    assert history == DurableHistory(last_completed_sha="s1", last_completed_at=NOW)


def test_apply_finish_without_a_publish_id_raises(config):
    state = waiting(phase=Phase.FINISHING, dispatch_sha="s1", dispatch_token="t1")
    with pytest.raises(ValueError, match="publish id"):
        apply(state, Finish("s1", "t1"), None, NOW, config)


def test_apply_mark_failed_settles_in_failed(config):
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1", attempt_count=1)
    new_state, history = apply(state, MarkFailed("boom"), None, NOW, config)
    assert new_state.phase == Phase.FAILED
    assert new_state.attempt_count == 2
    assert new_state.last_error == "boom"
    assert history is None


# --- apply_error --------------------------------------------------------


def test_apply_error_on_submit_settles_in_failed():
    state = waiting(phase=Phase.DISPATCHED, dispatch_sha="s1", attempt_count=0)
    new_state = apply_error(state, Submit("s1"), "connection refused")
    assert new_state.phase == Phase.FAILED
    assert new_state.attempt_count == 1
    assert new_state.last_error == "connection refused"


def test_apply_error_on_finish_settles_in_failed():
    state = waiting(
        phase=Phase.FINISHING, dispatch_sha="s1", dispatch_token="t1", attempt_count=1
    )
    new_state = apply_error(state, Finish("s1", "t1"), "gh: rate limited")
    assert new_state.phase == Phase.FAILED
    assert new_state.attempt_count == 2


@pytest.mark.parametrize("action", [Sleep(), MarkFailed("boom")])
def test_apply_error_rejects_actions_with_no_side_effect(action):
    state = waiting()
    with pytest.raises(ValueError, match="no side effect"):
        apply_error(state, action, "should never happen")
