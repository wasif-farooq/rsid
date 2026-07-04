"""RSI and swing-pivot (HH/HL/LH/LL) computation.

Two flavors of every stateful computation are provided:
  - a vectorized batch function operating on a full pandas Series
    (used by scripts/compute_features.py), and
  - an incremental class that consumes one bar at a time (used by
    scripts/infer_stream.py for live data).

Both flavors implement the *same* math so training-time features and live
inference features never drift apart.

Pivot confirmation is inherently lagged: a bar can only be confirmed as a
swing high/low once `lookback` bars have printed after it. Both APIs below
report a pivot's `pivot_iloc` (the bar that was the actual extreme) *and*
its `confirmed_iloc` (the first bar at which that fact is knowable,
`pivot_iloc + lookback`). Anything downstream that cares about "what could
a live system have known at this point" must key off `confirmed_iloc`, not
`pivot_iloc` — otherwise batch-computed features leak future information
that a streaming consumer would never have had.
"""

from collections import deque

import numpy as np
import pandas as pd

HH, HL, LH, LL = "HH", "HL", "LH", "LL"


# --------------------------------------------------------------------------
# RSI (Wilder smoothing)
# --------------------------------------------------------------------------

def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI. First `period` values are NaN (warm-up).

    Implemented as a thin wrapper around IncrementalWilderRSI (rather than
    a vectorized pandas.ewm formula) so batch-computed RSI is *exactly*
    identical to what the streaming detector sees bar-by-bar -- pandas'
    ewm(adjust=False) recursion seeds differently than classic Wilder
    smoothing and would otherwise drift a few tenths of a point off the
    streaming values, which is enough to occasionally flip which bar wins
    a pivot comparison.
    """
    rsi_calc = IncrementalWilderRSI(period=period)
    values = [rsi_calc.update(c) for c in close.to_numpy(dtype=float)]
    return pd.Series(values, index=close.index, dtype=float)


class IncrementalWilderRSI:
    """Streaming Wilder RSI, one close price at a time."""

    def __init__(self, period: int = 14):
        self.period = period
        self._prev_close = None
        self._avg_gain = None
        self._avg_loss = None
        self._count = 0
        self._seed_gains = []
        self._seed_losses = []

    def update(self, close: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            return None

        change = close - self._prev_close
        self._prev_close = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._count += 1

        if self._avg_gain is None:
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            if self._count < self.period:
                return None
            self._avg_gain = sum(self._seed_gains) / self.period
            self._avg_loss = sum(self._seed_losses) / self.period
        else:
            alpha = 1.0 / self.period
            self._avg_gain = (1 - alpha) * self._avg_gain + alpha * gain
            self._avg_loss = (1 - alpha) * self._avg_loss + alpha * loss

        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


# --------------------------------------------------------------------------
# Pivot (swing high/low) detection
# --------------------------------------------------------------------------

def find_pivots(values: pd.Series, lookback: int = 5):
    """Vectorized batch pivot detection.

    A bar at position i is a pivot high if values[i] is the strict max
    within [i-lookback, i+lookback]; symmetric for pivot low. Returns two
    lists of dicts (highs, lows), each with keys:
      pivot_iloc      - positional index of the extreme bar
      confirmed_iloc  - pivot_iloc + lookback (first bar it's knowable at)
      value           - values at pivot_iloc

    Rolling max/min narrows candidates to O(n); a window slice per
    candidate then enforces strict (unique) extremum. Rolling's default
    min_periods=window also means any NaN in a bar's window disqualifies
    it, matching the old bar-by-bar NaN check.
    """
    n = len(values)
    window_size = 2 * lookback + 1
    arr = values.to_numpy(dtype=float)
    roll_max = values.rolling(window_size, center=True).max().to_numpy()
    roll_min = values.rolling(window_size, center=True).min().to_numpy()

    valid_lo, valid_hi = lookback, n - lookback  # exclusive upper bound
    high_cands = np.where(arr == roll_max)[0]
    high_cands = high_cands[(high_cands >= valid_lo) & (high_cands < valid_hi)]
    low_cands = np.where(arr == roll_min)[0]
    low_cands = low_cands[(low_cands >= valid_lo) & (low_cands < valid_hi)]

    highs, lows = [], []
    for i in high_cands:
        center = arr[i]
        w = arr[i - lookback : i + lookback + 1]
        if np.sum(w == center) == 1:
            highs.append({"pivot_iloc": int(i), "confirmed_iloc": int(i) + lookback, "value": center})
    for i in low_cands:
        center = arr[i]
        w = arr[i - lookback : i + lookback + 1]
        if np.sum(w == center) == 1:
            lows.append({"pivot_iloc": int(i), "confirmed_iloc": int(i) + lookback, "value": center})

    return highs, lows


def label_pivot_events(events: list[dict], kind: str) -> list[dict]:
    """Attach HH/HL/LH/LL labels by comparing each pivot to the previous one.

    `kind` is "high" (produces HH/LH) or "low" (produces HL/LL). Mutates
    and returns the same list of dicts (adds a "label" key; first event has
    label=None since there's nothing to compare against).
    """
    prev_value = None
    for event in events:
        if prev_value is None:
            event["label"] = None
        elif kind == "high":
            event["label"] = HH if event["value"] > prev_value else LH
        else:
            event["label"] = HL if event["value"] > prev_value else LL
        prev_value = event["value"]
    return events


class IncrementalPivotDetector:
    """Streaming pivot detector using a sliding buffer of 2*lookback+1 bars.

    Call `update(value)` on each new bar (values must be fed in order, one
    per bar, no gaps skipped). Returns a dict describing a confirmed pivot
    at the *center* of the buffer (lagged by `lookback` bars), or None if
    nothing confirmed this step. Keys: kind ("high"/"low"), value, label
    (HH/HL/LH/LL, or None if this is the first pivot of its type seen),
    bars_ago (== lookback; how many updates back the actual extreme bar
    was, for the caller to resolve its timestamp from its own buffer).
    """

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self._buffer = deque(maxlen=2 * lookback + 1)
        self._prev_high = None
        self._prev_low = None

    def update(self, value: float):
        self._buffer.append(value)
        if len(self._buffer) < self._buffer.maxlen:
            return None

        window = list(self._buffer)
        center = window[self.lookback]
        result = None

        if center == max(window) and window.count(center) == 1:
            label = None
            if self._prev_high is not None:
                label = HH if center > self._prev_high else LH
            self._prev_high = center
            result = {"kind": "high", "value": center, "label": label, "bars_ago": self.lookback}

        if center == min(window) and window.count(center) == 1:
            label = None
            if self._prev_low is not None:
                label = HL if center > self._prev_low else LL
            self._prev_low = center
            if result is None:
                result = {"kind": "low", "value": center, "label": label, "bars_ago": self.lookback}

        return result
