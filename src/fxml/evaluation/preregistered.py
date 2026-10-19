"""The pass marks from docs/preregistration-daily-trend.md, written as code so they are applied
literally. Changing a number here after a run defeats the point of writing it down first."""

DEV_SHARPE_MIN = 0.40
CONFIRM_SHARPE_MIN = 0.30
STRESS_SHARPE_MIN = 0.15
MAX_DRAWDOWN_LIMIT = -0.25          # on the cumulative sum of daily log returns
VOL_TARGET_SHARPE_GAIN = 0.05       # over the equal-notional baseline...
VOL_TARGET_DRAWDOWN_GAIN = 0.03     # ...or a max drawdown this much smaller


def development_verdict(targeted: dict, baseline: dict) -> dict:
    """Bars 1, 4 and 5 on the development period. Both dicts hold sharpe and max_drawdown."""
    verdict = {
        "sharpe": targeted["sharpe"] >= DEV_SHARPE_MIN,
        "drawdown": targeted["max_drawdown"] >= MAX_DRAWDOWN_LIMIT,
        "vol_targeting": (targeted["sharpe"] - baseline["sharpe"] >= VOL_TARGET_SHARPE_GAIN
                          or targeted["max_drawdown"] - baseline["max_drawdown"]
                          >= VOL_TARGET_DRAWDOWN_GAIN),
    }
    verdict["all"] = all(verdict.values())
    return verdict


def confirmation_verdict(normal: dict, stressed: dict) -> dict:
    """Bars 2, 3 and 4 on the confirmation period: normal costs, then doubled costs."""
    verdict = {
        "sharpe": normal["sharpe"] >= CONFIRM_SHARPE_MIN,
        "stress": stressed["sharpe"] >= STRESS_SHARPE_MIN,
        "drawdown": normal["max_drawdown"] >= MAX_DRAWDOWN_LIMIT,
    }
    verdict["all"] = all(verdict.values())
    return verdict
