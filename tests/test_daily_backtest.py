"""Tests for the daily trading engine: timing and costs."""
import numpy as np
import pandas as pd
import pytest

from fxml.evaluation.daily_backtest import (fill_spread_fraction, net_daily_pnl,
                                            run_daily_portfolio)



def days(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC", name="time")
    return pd.Series(values, index=index, dtype=float)


OPENS = days([100, 110, 121, 133.1, 146.41])  # the open rises 10% every day
FREE = days([0.0] * 5)


def test_a_position_decided_today_earns_from_tomorrows_open_to_the_day_after():
    position = days([0, 1, 0, 0, 0])  # decided at the close of day 1
    pnl = net_daily_pnl(position, OPENS, FREE)
    # filled at day 2's open (121) and held to day 3's open (133.1); nothing earned before that
    assert pnl["gross"].iloc[:4].to_numpy() == pytest.approx([0, 0, np.log(1.1), 0])


def test_changing_the_position_pays_half_the_spread_per_unit_traded():
    position = days([0, 2, 0, 0, 0])  # in at 2x after day 1, out after day 2
    pnl = net_daily_pnl(position, OPENS, days([0.0004] * 5))  # a spread of 4 bps of price
    assert pnl["cost"].iloc[:4].to_numpy() == pytest.approx([0, 0, 0.0004, 0.0004])


def test_flipping_from_long_to_short_trades_two_units():
    position = days([0, 1, -1, -1, -1])
    pnl = net_daily_pnl(position, OPENS, days([0.0004] * 5))
    assert pnl["cost"].iloc[3] == pytest.approx(2 * pnl["cost"].iloc[2])


def test_net_is_gross_minus_cost():
    position = days([0, 1, 1, 1, 1])
    pnl = net_daily_pnl(position, OPENS, days([0.0004] * 5))
    assert pnl["net"].iloc[2] == pytest.approx(np.log(1.1) - 0.0002)  # earns, pays on entry
    assert pnl["net"].iloc[3] == pytest.approx(np.log(1.1))  # holds: no new cost


def test_a_missing_position_counts_as_flat():
    position = days([np.nan, np.nan, 1, 1, 1])
    pnl = net_daily_pnl(position, OPENS, FREE)
    assert not pnl.iloc[:4].isna().any().any()
    assert pnl["held"].iloc[:3].tolist() == [0, 0, 0]


def test_the_last_day_has_no_next_open_so_it_has_no_return():
    pnl = net_daily_pnl(days([1, 1, 1, 1, 1]), OPENS, FREE)
    assert np.isnan(pnl["net"].iloc[-1])


def test_rewriting_later_opens_cannot_change_earlier_pnl():
    position = days([1, -1, 1, 1, -1, 1, 1, 1])
    opens = days(np.linspace(100, 108, 8))
    changed = opens.copy()
    changed.iloc[5:] = [50, 60, 70]
    spread = days([0.0002] * 8)
    pd.testing.assert_frame_equal(net_daily_pnl(position, opens, spread).iloc[:4],
                                  net_daily_pnl(position, changed, spread).iloc[:4])


def test_a_missing_spread_is_an_error_not_a_free_trade():
    spread = days([0.0004, np.nan, 0.0004, 0.0004, 0.0004])
    with pytest.raises(ValueError):
        net_daily_pnl(days([0, 1, 0, 0, 0]), OPENS, spread)

RECORDED_FROM = pd.Timestamp("2015-01-01", tz="UTC")


def daily_frame(open_spread, open_=1.0) -> pd.DataFrame:
    """Six days from 29 Dec 2014: three before the recorded-spread cutoff, three after."""
    index = pd.date_range("2014-12-29", periods=len(open_spread), freq="D", tz="UTC", name="time")
    return pd.DataFrame({"open": open_, "open_spread": open_spread}, index=index)


def test_early_placeholder_spreads_are_replaced_by_the_recorded_median():
    fraction = fill_spread_fraction(daily_frame([99, 99, 99, 10, 20, 30]), 0.0001, RECORDED_FROM)
    assert fraction.to_numpy() == pytest.approx(np.array([20, 20, 20, 10, 20, 30]) * 0.0001)


def test_a_recorded_spread_of_zero_is_treated_as_missing_not_free():
    fraction = fill_spread_fraction(daily_frame([99, 99, 99, 10, 0, 30]), 0.0001, RECORDED_FROM)
    assert fraction.to_numpy() == pytest.approx(np.array([20, 20, 20, 10, 20, 30]) * 0.0001)


def test_the_spread_fraction_shrinks_as_the_price_rises():
    cheap = fill_spread_fraction(daily_frame([5] * 6, open_=1.0), 0.0001, RECORDED_FROM)
    dear = fill_spread_fraction(daily_frame([5] * 6, open_=2.0), 0.0001, RECORDED_FROM)
    assert dear.iloc[-1] == pytest.approx(cheap.iloc[-1] / 2)


def test_the_cost_multiplier_scales_every_fraction():
    base = fill_spread_fraction(daily_frame([10] * 6), 0.0001, RECORDED_FROM)
    doubled = fill_spread_fraction(daily_frame([10] * 6), 0.0001, RECORDED_FROM, cost_multiplier=2.0)
    assert doubled.to_numpy() == pytest.approx(2 * base.to_numpy())

EARLY = pd.Timestamp("2000-01-01", tz="UTC")  # every synthetic day counts as recorded
POINTS = {"AAA": 0.0001, "BBB": 0.0001, "CCC": 0.0001}


def pair_from_close(close: np.ndarray) -> pd.DataFrame:
    """Daily bars where each open is the previous close (no gaps) and the spread is 10 points."""
    index = pd.date_range("2010-01-01", periods=len(close), freq="B", tz="UTC", name="time")
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close, "open_spread": 10.0},
                        index=index)


def random_pair(seed: int, n: int = 700) -> pd.DataFrame:
    steps = np.random.default_rng(seed).normal(0.0005, 0.006, n)
    return pair_from_close(1.2 * np.exp(np.cumsum(steps)))


def random_pairs(n: int = 700) -> dict:
    return {name: random_pair(seed, n) for seed, name in enumerate(POINTS)}


def test_before_any_signal_exists_the_portfolio_is_flat():
    result = run_daily_portfolio(random_pairs(), POINTS, EARLY)
    assert (result.daily["net"].iloc[:100] == 0).all()
    assert (result.daily["exposure"].iloc[:100] == 0).all()


def test_equal_notional_splits_the_capital_across_pairs():
    steady = {name: pair_from_close(1.2 * np.exp(0.001 * np.arange(400))) for name in ["A", "B"]}
    result = run_daily_portfolio(steady, {"A": 0.0001, "B": 0.0001}, EARLY, sizing="equal_notional")
    assert result.daily["exposure"].iloc[300] == pytest.approx(1.0)  # two pairs at 1/2 each
    assert result.daily["gross"].iloc[300] == pytest.approx(0.001)  # both earn 0.001 a day


def test_volatility_targeting_brings_realised_volatility_near_the_target():
    result = run_daily_portfolio(random_pairs(1500), POINTS, EARLY)
    realised = result.daily["gross"].iloc[400:-1].std() * np.sqrt(260)
    assert 0.07 < realised < 0.13


def test_the_multiplier_does_not_explode_when_signals_first_appear():
    result = run_daily_portfolio(random_pairs(1500), POINTS, EARLY)
    assert result.daily["exposure"].max() < 10 * result.daily["exposure"].iloc[400:].median()


def test_rewriting_the_future_cannot_change_earlier_portfolio_returns():
    original = random_pairs()
    rng = np.random.default_rng(9)
    wild = np.exp(np.cumsum(rng.normal(0.01, 0.05, 200)))  # a very different path from day 500
    changed = {}
    for name, bars in original.items():
        bars = bars.copy()
        bars.iloc[500:, bars.columns.get_indexer(["open", "close"])] *= wild[:, None]
        changed[name] = bars
    before = run_daily_portfolio(original, POINTS, EARLY).daily
    after = run_daily_portfolio(changed, POINTS, EARLY).daily
    pd.testing.assert_frame_equal(before.iloc[:499], after.iloc[:499])
    # the position held on day 499 was decided before the rewrite, so it cannot have changed either
    pd.testing.assert_series_equal(before["exposure"].iloc[:500], after["exposure"].iloc[:500])


def test_stress_costs_are_exactly_double_and_gross_is_unchanged():
    base = run_daily_portfolio(random_pairs(), POINTS, EARLY).daily
    stress = run_daily_portfolio(random_pairs(), POINTS, EARLY, cost_multiplier=2.0).daily
    assert stress["cost"].iloc[:-1].to_numpy() == pytest.approx(2 * base["cost"].iloc[:-1].to_numpy())
    assert stress["gross"].iloc[:-1].to_numpy() == pytest.approx(base["gross"].iloc[:-1].to_numpy())


def test_an_unknown_sizing_is_rejected():
    with pytest.raises(ValueError):
        run_daily_portfolio(random_pairs(), POINTS, EARLY, sizing="martingale")
