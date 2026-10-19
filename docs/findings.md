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

## Breadth test across pairs

Idea: one pair's edge is about the size of its cost, but many pairs with partly independent edges
might diversify it into something tradable.

- Data: 11 pairs downloaded from the broker (EURCHF has no H1 data there). Point sizes are now set
  per symbol in `config.POINT_SIZES`, because the JPY pairs quote 3 digits.
- **Cost screen, fixed before any training** and using only cost and volatility: median round-trip
  spread divided by median H1 ATR(14), 2015 onward. Keep pairs at or below 1.5 times EURUSD's ratio
  (limit 0.0795).

| Kept | Cost / range | Dropped | Cost / range |
|---|---|---|---|
| EURUSD | 0.053 | AUDUSD | 0.084 |
| GBPUSD | 0.056 | USDCHF | 0.086 |
| EURJPY | 0.061 | AUDJPY | 0.095 |
| GBPJPY | 0.064 | NZDUSD | 0.120 |
| USDJPY | 0.070 | EURGBP | 0.137 |
| USDCAD | 0.076 | | |

- Method: the exact horizon-12 setup of the EURUSD confirmation, one run per kept pair, recorded
  spreads, no tuning. Pass bars written in advance: equal-weight portfolio net Sharpe above 0.6, at
  least 4 of 6 pairs with positive net Sharpe, and a positive mean net return per trade with 0.3 bps
  of extra slippage.

| Pair | Gross bps/trade | Net bps/trade | Net with slippage | Net Sharpe | Mean rank corr |
|---|---|---|---|---|---|
| EURUSD | +1.02 | +0.34 | +0.04 | +0.33 | +0.029 |
| GBPUSD | -2.07 | -2.93 | -3.23 | -2.15 | +0.006 |
| EURJPY | +2.48 | +1.31 | +1.01 | +0.98 | +0.020 |
| GBPJPY | +1.54 | +0.17 | -0.13 | +0.12 | +0.016 |
| USDJPY | +1.22 | +0.04 | -0.26 | +0.03 | +0.008 |
| USDCAD | -0.76 | -1.88 | -2.18 | -2.20 | -0.005 |

Portfolio net Sharpe **-0.82** (bar: above 0.6, failed); 4 of 6 pairs with positive net Sharpe (bar:
at least 4, passed, though two of them are near zero); mean net return with slippage **-0.79 bps**
per trade (bar: above 0, failed). **Overall: failed.**

- EURUSD reproduced the earlier confirmation result (net Sharpe 0.33), a useful consistency check.
- The six strategies' returns correlate at only 0.0 to 0.38 (mostly about 0.05), so the breadth was
  real; there was simply no average edge to diversify.
- The spread of results (-2.2 to +1.0 Sharpe) is consistent with noise around a mean near zero. As a
  rough estimate each pair's average gross return carries an error margin of about 1.2 bps; the best
  pair (EURJPY) is about two margins above zero and the worst (GBPUSD) about two below, which is
  ordinary for the best and worst of six. Picking EURJPY afterwards would be choosing the lucky one.

**Conclusion:** no tradable direction edge from these features on H1 across six pairs. By the pre-set
rule, the direction search on public price data stops here.

## Where my own predictions were wrong

- Expected the edge to shrink at longer horizons; horizon 12 beat horizon 4 on the development era
  (then shrank again on the confirmation era).
- Expected persistence to score 0.2-0.3 for volatility; it scored -0.03.
- Expected time-of-day to correlate with 24-bar volatility; it does not, because the 24-bar window
  spans every hour.
- Expected HAR+ to add at most +0.02 over HAR; it added +0.036.
- Expected EURUSD's cost to be about 10% of its hourly range; it is 5.3% (median hourly ATR is 12 bps,
  not 8-10). That also made my gold comparison too generous (about 3 times cheaper, not 5).
- Expected JPY pairs to cost less relative to their range than EURUSD; all are higher, so EURUSD is
  the cheapest pair on both measures.
- Expected the six pairs' strategy returns to correlate at 0.3-0.6; they correlate at 0.0-0.38.
- Expected a portfolio Sharpe of 0.3-0.6; it was -0.82.

## Caveats

- Demo account: real fills would be worse (slippage, requotes), and swap/rollover charges are not
  modelled. Treat a positive backtest as an upper bound and a negative one as likely true.
- Costs are the recorded spread only. Commission-based accounts price differently.
- The 2015-2026 test windows have been inspected repeatedly, for EURUSD and now for six pairs. Any
  further model or rule change needs genuinely new data, such as bars that arrive after 2026-10-19.
- One seed per run; no confidence intervals beyond the rough noise bands in the charts.

## Reproduce

All runs are stored in `data/forex.db` (`runs` and `fold_metrics`): `lstm-h4-baseline`, `dev-h4`,
`dev-h12`, `dev-h24`, `confirm-h12` and `breadth-h12-<PAIR>` for the six pairs (plus any later runs).
See the README for commands; the horizon study, volatility work and breadth test live in notebooks
04-09. `scripts/probe_pairs.py` lists the pairs the broker offers with their history and cost.

## Ideas not yet tried

- Volatility-targeted position sizing (a new pre-registered test, development era first).
- An LSTM for volatility, which must beat HAR+ by at least +0.02 R2 and win in at least 9 of 13 folds.
- New information sources (other pairs' returns, rates, macro data) as a clearly new hypothesis,
  confirmed only on data that arrives after 2026-10-19.
- Lower-frequency strategies (daily or weekly trend or carry), which amortise cost over larger moves.
- A forward paper test of the horizon-12 model on bars that arrive from now on.
