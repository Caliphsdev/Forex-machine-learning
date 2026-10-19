# Pre-registration: daily trend, volatility-targeted

Written 2026-10-19, before any result. Pass marks are not edited after a run; any change is
logged at the bottom.

## Question
Does a fixed-rule trend strategy on daily bars across 11 pairs make money after costs?

## Data and days
- H1 bars for the 11 pairs in data/forex.db (EURCHF has no data at this broker).
- An FX day runs 17:00 to 17:00 New York (broker midnight). Daily bars come from
  `fxml.features.resample.to_daily`.
- Every day is kept. Only the final day of the data is dropped if it has under 20 H1 bars.
- A pair with no bar on a given day holds no position that day.

## Strategy (all settings fixed, nothing fitted)
- Signal: mean of the signs of the trailing log return over 21, 63 and 126 days, at the close.
- Pair size: signal x 10% / annualised volatility. Volatility is the EWMA of squared daily
  log returns, span 60 days, annualised with 260 days.
- Portfolio: sum of pair exposures, multiplied by 10% / annualised EWMA volatility (span 60)
  of the unscaled portfolio's gross daily return, using returns up to the same close.
  No leverage cap; maximum leverage is reported.
- Timing: decided at the close of day t, filled at the open of day t+1, held to the open of
  day t+2.

## Costs
- Half the spread for every unit of position change, charged at the fill bar (the day's
  first H1 bar, the 17:00 New York roll).
- 2015 onward: that bar's recorded spread. Before 2015: a flat per-pair spread equal to the
  median 2015-onward spread of that same bar (earlier recorded spreads are placeholders).
- Stress test: costs doubled.
- Not modelled: swap/rollover, commission, interest carry.

## Periods
- Development: 2001-01-01 to 2014-12-31 (1999-2000 is warm-up).
- Confirmation: 2015-01-01 to 2026-10-16, run once.
- Forward: bars after 2026-10-16.

## Pass marks (all must hold)
Sharpe = mean / std of daily net returns x sqrt(260), no risk-free rate. Drawdown is on the
cumulative sum of daily log returns.
1. Development: portfolio net Sharpe >= 0.40.
2. Confirmation: portfolio net Sharpe >= 0.30.
3. Confirmation with doubled costs: net Sharpe >= 0.15.
4. Max drawdown no worse than -25% in both periods.
5. Development: the volatility-targeted portfolio beats the same signals at equal notional
   (signal / 11 per pair, no volatility scaling) by at least +0.05 Sharpe, or has a max
   drawdown at least 3 points smaller.
Reported but not a bar: block-bootstrap confidence interval on each Sharpe, turnover,
maximum leverage, per-pair results.

## Predictions
Development Sharpe 0.3-0.6; confirmation 0.0-0.3; about 25% chance of passing every bar.

## Limits
Fixed rules make "development" and "confirmation" both effectively out of sample, and the
2015-2026 prices were examined for the H1 work, so confirmation is weaker than fresh data.
The forward bars are the real test. Demo-account spreads; positive results are an upper bound.

## Changes after approval (made while building, before any result)
1. "60-day EWMA" means span 60; a year is 260 days.
2. Costs use the fill bar's spread, not the day's average. On 2015+ data that bar's median
   spread is 1.3x (EURUSD) to 2.8x (GBPJPY) the all-day median.
3. Portfolio volatility scaling is specified as above.
4. Only a final incomplete day is dropped.
