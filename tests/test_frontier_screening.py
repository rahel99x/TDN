"""A failed or missing teacher cannot manufacture classical headroom."""
from tdn.analysis.frontier.screening import summarize_headroom


def row(parent="easy", **kwargs):
    value = dict(parent_id=parent, field_cluster=parent, grid=8, track="discrete", final_time=.1,
        family="df", steps=1, upper_rms=1e-7, upper_max=1e-7, reference_accepted=True,
        regime="normal", numerical_failure=False, cost_seconds=.1)
    return {**value, **kwargs}


def test_easy_cases_retained_not_eligible_for_solver_claim():
    result = summarize_headroom([row(), row(family="etdrk4", steps=4)], 1e-5)
    assert not result["solver_utility_eligible"]
    assert result["neural_research_allowed"]
    assert result["strata"]["normal"]["easy"] == 1


def test_missing_teacher_and_infeasible_rows_never_create_headroom():
    result = summarize_headroom([row(reference_accepted=False), row("hard", upper_max=1.)], 1e-5)
    assert result["conditions"] == 2
    assert result["headroom_conditions"] == 0
    assert result["strata"]["normal"]["unresolved"] == 1
    assert result["strata"]["normal"]["infeasible"] == 1


def test_headroom_needs_strong_feasible_solver_at_identical_final_time():
    rows = [row(upper_max=1.), row(family="etdrk4", steps=2), row(final_time=.3, upper_max=1.)]
    result = summarize_headroom(rows, 1e-5)
    assert result["headroom_conditions"] == 1
    assert result["solver_utility_eligible"]
    assert result["conditions"] == 2
    assert result["selection_split"] == "development"


def test_joint_maximum_target_cannot_pass_on_small_rms_alone():
    result = summarize_headroom([row(upper_rms=1e-10, upper_max=.01)], 1e-5)
    assert not result["solver_utility_eligible"]
    assert result["strata"]["normal"]["infeasible"] == 1
