"""Tests that the pre-registered pass marks are applied literally."""
from fxml.evaluation.preregistered import confirmation_verdict, development_verdict


def stats(sharpe: float, drawdown: float = -0.10) -> dict:
    return {"sharpe": sharpe, "max_drawdown": drawdown}


def test_development_sharpe_must_reach_040():
    baseline = stats(0.0)
    assert development_verdict(stats(0.40), baseline)["sharpe"]
    assert not development_verdict(stats(0.39), baseline)["sharpe"]


def test_drawdown_must_be_no_worse_than_minus_25_percent():
    baseline = stats(0.0)
    assert development_verdict(stats(0.5, -0.25), baseline)["drawdown"]
    assert not development_verdict(stats(0.5, -0.26), baseline)["drawdown"]


def test_volatility_targeting_passes_on_a_sharpe_gain_of_005():
    assert development_verdict(stats(0.60), stats(0.50))["vol_targeting"]
    assert not development_verdict(stats(0.52), stats(0.50))["vol_targeting"]


def test_volatility_targeting_also_passes_on_a_drawdown_3_points_smaller():
    assert development_verdict(stats(0.50, -0.10), stats(0.50, -0.14))["vol_targeting"]
    assert not development_verdict(stats(0.50, -0.10), stats(0.50, -0.12))["vol_targeting"]


def test_development_passes_only_when_every_bar_passes():
    assert development_verdict(stats(0.60, -0.10), stats(0.40, -0.10))["all"]
    assert not development_verdict(stats(0.30, -0.10), stats(0.10, -0.10))["all"]


def test_confirmation_needs_030_normally_and_015_with_doubled_costs():
    ok = confirmation_verdict(stats(0.30), stats(0.15))
    assert ok["sharpe"] and ok["stress"] and ok["all"]
    assert not confirmation_verdict(stats(0.29), stats(0.20))["sharpe"]
    assert not confirmation_verdict(stats(0.40), stats(0.14))["stress"]


def test_confirmation_drawdown_is_checked_on_the_normal_run():
    assert not confirmation_verdict(stats(0.5, -0.30), stats(0.3))["drawdown"]
