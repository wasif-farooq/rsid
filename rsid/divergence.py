"""Hidden RSI divergence detection, shared by batch labeling and live streaming.

Pivots are found on the RSI line itself (the oscillator), then the price at
that same pivot bar is compared against price at the previous RSI pivot of
the same kind. This is the standard definition used by most RSI-divergence
indicators:

  Hidden bullish: RSI makes a Lower Low (LL) while price makes a Higher Low
                  (HL) at the same two swing points -> uptrend continuation.
  Hidden bearish: RSI makes a Higher High (HH) while price makes a Lower
                  High (LH) at the same two swing points -> downtrend
                  continuation.

Both the batch and streaming detectors tag an event at its *confirmation*
point (pivot bar + lookback), not at the pivot bar itself, since that's the
earliest point a real-time system could have known about it. Training data
built from the batch path and signals produced by the streaming path are
therefore directly comparable.
"""

from collections import deque

from .indicators import HH, HL, LH, LL, IncrementalPivotDetector, find_pivots, label_pivot_events

HIDDEN_BULLISH = "hidden_bullish"
HIDDEN_BEARISH = "hidden_bearish"


def detect_hidden_divergence_batch(df, rsi_col="rsi", low_col="low", high_col="high", lookback=5):
    """Returns a list of divergence event dicts, sorted by confirmed_iloc.

    Each event: type, pivot_iloc, confirmed_iloc, rsi_value, price_value,
    prev_rsi_value, prev_price_value.
    """
    highs, lows = find_pivots(df[rsi_col], lookback)
    highs = label_pivot_events(highs, kind="high")
    lows = label_pivot_events(lows, kind="low")

    events = []

    prev_price_low = None
    prev_rsi_low = None
    for piv in lows:
        price_v = df[low_col].iloc[piv["pivot_iloc"]]
        if piv["label"] == LL and prev_price_low is not None and price_v > prev_price_low:
            events.append(
                {
                    "type": HIDDEN_BULLISH,
                    "pivot_iloc": piv["pivot_iloc"],
                    "confirmed_iloc": piv["confirmed_iloc"],
                    "rsi_value": piv["value"],
                    "price_value": price_v,
                    "prev_rsi_value": prev_rsi_low,
                    "prev_price_value": prev_price_low,
                }
            )
        prev_price_low = price_v
        prev_rsi_low = piv["value"]

    prev_price_high = None
    prev_rsi_high = None
    for piv in highs:
        price_v = df[high_col].iloc[piv["pivot_iloc"]]
        if piv["label"] == HH and prev_price_high is not None and price_v < prev_price_high:
            events.append(
                {
                    "type": HIDDEN_BEARISH,
                    "pivot_iloc": piv["pivot_iloc"],
                    "confirmed_iloc": piv["confirmed_iloc"],
                    "rsi_value": piv["value"],
                    "price_value": price_v,
                    "prev_rsi_value": prev_rsi_high,
                    "prev_price_value": prev_price_high,
                }
            )
        prev_price_high = price_v
        prev_rsi_high = piv["value"]

    events.sort(key=lambda e: e["confirmed_iloc"])
    return events


class StreamingDivergenceDetector:
    """Causal, one-bar-at-a-time counterpart to detect_hidden_divergence_batch.

    Call update(timestamp, high, low, close, rsi) for every new confirmed
    bar, in order. Returns a divergence event dict (same "type"/value keys
    as the batch path, plus timestamp/confirmed_timestamp) or None.
    """

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self._rsi_pivots = IncrementalPivotDetector(lookback)
        self._bar_buffer = deque(maxlen=2 * lookback + 1)
        self._prev_price_low = None
        self._prev_rsi_low = None
        self._prev_price_high = None
        self._prev_rsi_high = None

    def update(self, timestamp, high: float, low: float, close: float, rsi: float | None):
        self._bar_buffer.append({"timestamp": timestamp, "high": high, "low": low, "close": close})

        if rsi is None:
            return None
        pivot = self._rsi_pivots.update(rsi)
        if pivot is None or len(self._bar_buffer) < self._bar_buffer.maxlen:
            return None

        pivot_bar = self._bar_buffer[self.lookback]
        result = None

        if pivot["kind"] == "low":
            price_v = pivot_bar["low"]
            if (
                pivot["label"] == LL
                and self._prev_price_low is not None
                and price_v > self._prev_price_low
            ):
                result = {
                    "type": HIDDEN_BULLISH,
                    "timestamp": pivot_bar["timestamp"],
                    "confirmed_timestamp": timestamp,
                    "rsi_value": pivot["value"],
                    "price_value": price_v,
                    "prev_rsi_value": self._prev_rsi_low,
                    "prev_price_value": self._prev_price_low,
                }
            self._prev_price_low = price_v
            self._prev_rsi_low = pivot["value"]
        else:
            price_v = pivot_bar["high"]
            if (
                pivot["label"] == HH
                and self._prev_price_high is not None
                and price_v < self._prev_price_high
            ):
                result = {
                    "type": HIDDEN_BEARISH,
                    "timestamp": pivot_bar["timestamp"],
                    "confirmed_timestamp": timestamp,
                    "rsi_value": pivot["value"],
                    "price_value": price_v,
                    "prev_rsi_value": self._prev_rsi_high,
                    "prev_price_value": self._prev_price_high,
                }
            self._prev_price_high = price_v
            self._prev_rsi_high = pivot["value"]

        return result
