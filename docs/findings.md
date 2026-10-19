# Findings

Question: can an LSTM trained on MT5 history find a tradable edge in EURUSD, and is volatility
usable as a helper? Everything below was measured out of sample with walk-forward validation and
judged against pass marks written down before each experiment ran. Dates: work finished 2026-10-19.

## Data

- OctaFX-Demo MT5, EURUSD H1: 172,606 bars from 1999-01-04 to 2026-10-16. Real data starts in
  1999 (the euro's launch); the broker's earlier "history" is one synthetic bar per day.
- 1999 onward has about 6,150-6,290 bars per year. The broker clock is New York time + 7 hours.
- **Before 2015 the data is less trustworthy:** weekend gaps are irregular in 2007-2014 (timestamps
  can be off by up to an hour) and the spread column looks like placeholder values. Training used
  2015 onward with recorded spreads. The 2001-2014 era served only as a development set, with a flat
  0.8 bps cost.
- Average recorded cost in 2015-2026 was about 0.76 bps per trade (spread only, no commission).

## Method

- Walk-forward folds: expanding training window (first span 31,000 rows, about 5 years), 3,100-row
  test windows (about 6 months), a gap equal to the label horizon, and a 3,000-row validation tail
  for early stopping. 13 folds on 2015-2026.
- Direction model: one LSTM layer of 64 units, dropout 0.2, 48-bar lookback, two heads (standardised
  forward return, up/down logit), about 19.8k parameters. Eleven scale-free features (returns,
  log ATR, log range, log volatility, relative tick volume, cyclical hour and weekday).
- Trading rule: take the most confident 20% of signals, hold for the label horizon, charge the spread
  per trade. Positions overlap, each sized 1/horizon.
- Leakage controls (each backed by tests): chronological splits with a purge gap, scaler and target
  standardiser fitted on training rows only, a test showing that rewriting the test window cannot
  change training history, and indicator tests showing no feature looks ahead.

## Direction results

**Horizon 4, 2015-2026 (13 folds):** mean accuracy 0.514 against a 0.500 majority guess; beat the
guess in 10 of 13 folds; mean rank correlation +0.026, positive in 11 of 13 folds; error no better
than predicting zero. Accuracy rose with confidence (quintiles 0.493, 0.506, 0.513, 0.520, 0.540),
so the signal is real, but gross return in the top quintile was only 0.82 bps against a cost of
about 0.76 bps. After costs: 8,060 trades, +0.06 bps per trade, and -0.24 bps with 0.3 bps of extra
slippage.

**Horizon study on the development era (2001-2014, flat 0.8 bps cost), chosen by the pre-set rule:**

| Horizon | Mean rank corr | Positive folds | Net Sharpe | Eligible |
|---|---|---|---|---|
| 4 | +0.028 | 14 of 18 | 0.20 | yes |
| 12 | +0.019 | 11 of 17 | 0.83 | yes (winner) |
| 24 | +0.004 | 9 of 17 | -0.79 | no (under 60% of folds) |

**Confirmation of horizon 12 on 2015-2026 with recorded spreads (run once): failed.**

| Pre-set bar | Needed | Result |
|---|---|---|
| Net Sharpe | above 0.5 | 0.33 |
| Net bps per trade with 0.3 bps slippage | at least +0.5 | +0.04 |

Other numbers: gross +1.02 and net +0.34 bps per trade, rank correlation +0.029 (positive in 8 of
13 folds), max drawdown -7.0%, total return +2.3% over about six and a half years. The after-cost
equity curve peaked near +9% in early 2025 and gave most of it back in the spring 2025 shock.

**Conclusion:** a faint, repeatable directional signal exists on EURUSD H1, but it is about the same
size as the retail spread. It is not tradable on these features and costs.

## Volatility results

Target: log of the root-mean-square 1-bar return over the next 24 bars.

| Forecaster | Mean out-of-sample R2 (vs training mean) |
|---|---|
| Persistence (trailing 24-bar vol) | -0.030 |
| HAR (linear on 6/24/120/500-bar vol) | 0.350 |
| HAR+ (adds range, ATR, volume, time) | 0.386 |

HAR beat persistence in 13 of 13 folds (pre-set bar: 11 of 13 and R2 above 0.3: passed). HAR+ beat HAR
in 11 of 13 folds. Pooled forecast correlation was 0.651 over 40,300 out-of-sample bars. Volatility
is predictable, unlike direction; the long (monthly) window carries most of the information.

**Volatility filter on the development era (horizon 12):** trading only when forecast volatility was
in the top half raised net return per trade (1.16 to 1.46 bps) but lowered net Sharpe from 0.83 to
0.55 (bar: at least +0.20) and left the max drawdown unchanged at -11.2% (the 2008 crisis). The
low-volatility half had a drawdown of only -4.8%. **Failed.** Takeaway: volatility forecasts look
more useful for risk control than as a source of extra profit.

## Where my own predictions were wrong

- Expected the edge to shrink at longer horizons; horizon 12 beat horizon 4 on the development era
  (then shrank again on the confirmation era).
- Expected persistence to score 0.2-0.3 for volatility; it scored -0.03.
- Expected time-of-day to correlate with 24-bar volatility; it does not, because the 24-bar window
  spans every hour.
- Expected HAR+ to add at most +0.02 over HAR; it added +0.036.

## Caveats

- Demo account: real fills would be worse (slippage, requotes), and swap/rollover charges are not
  modelled. Treat a positive backtest as an upper bound and a negative one as likely true.
- Costs are the recorded spread only. Commission-based accounts price differently.
- The 2015-2026 test windows have been inspected repeatedly. Any further model or rule change needs
  genuinely new data, such as bars that arrive after 2026-10-16.
- One seed per run; no confidence intervals beyond the rough noise bands in the charts.

## Reproduce

All runs are stored in `data/forex.db` (`runs` and `fold_metrics`): `lstm-h4-baseline`, `dev-h4`,
`dev-h12`, `dev-h24` and `confirm-h12` (plus any later runs). See the README for commands; the
horizon study and volatility work live in notebooks 04-07.

## Ideas not yet tried

- Volatility-targeted position sizing (a new pre-registered test, development era first).
- An LSTM for volatility, which must beat HAR+ by at least +0.02 R2 and win in at least 9 of 13 folds.
- Other pairs or timeframes, each with fresh pre-set rules to limit multiple-testing luck.
- A forward paper test of the horizon-12 model on bars that arrive from now on.
