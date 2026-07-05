"""OHLCV bar sourcing for live trading: either bucketed from raw trade
prints (BarAggregator, 1s bars) or read directly off Binance's own
kline/candlestick stream (KlineBarTracker, any interval Binance supports
natively -- 1m is the smallest). Shared by scripts/infer_stream.py
(manual smoke-test) and scripts/run_paper_trading.py (production entrypoint).
"""

import threading


class BarAggregator:
    """Buckets raw trade prices into 1s OHLCV bars, gap-filling silent seconds."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_second = None
        self._bar = None
        self._last_close = None

    def on_trade(self, price: float, qty: float, ts_ms: int):
        second = ts_ms // 1000
        with self._lock:
            if self._current_second is None:
                self._current_second = second
                self._bar = {"open": price, "high": price, "low": price, "close": price, "volume": qty}
                return []
            if second == self._current_second:
                b = self._bar
                b["high"] = max(b["high"], price)
                b["low"] = min(b["low"], price)
                b["close"] = price
                b["volume"] += qty
                return []
            if second < self._current_second:
                return []  # out-of-order trade, ignore
            return self._advance_to(second, next_open=price, next_qty=qty)

    def flush_to(self, wall_second: int):
        """Force-finalize bars up to wall_second even with no new trade."""
        with self._lock:
            if self._current_second is None or wall_second <= self._current_second:
                return []
            return self._advance_to(wall_second, next_open=None, next_qty=None)

    def _advance_to(self, new_second: int, next_open, next_qty):
        """Caller must hold self._lock. Finalizes the buffered bar, gap-fills
        any skipped seconds, and starts a fresh bar at new_second."""
        finished = [(self._current_second, dict(self._bar))]
        self._last_close = self._bar["close"]

        gap_second = self._current_second + 1
        while gap_second < new_second:
            flat = {
                "open": self._last_close,
                "high": self._last_close,
                "low": self._last_close,
                "close": self._last_close,
                "volume": 0.0,
            }
            finished.append((gap_second, flat))
            gap_second += 1

        if next_open is not None:
            self._bar = {"open": next_open, "high": next_open, "low": next_open, "close": next_open, "volume": next_qty}
        else:
            self._bar = {
                "open": self._last_close,
                "high": self._last_close,
                "low": self._last_close,
                "close": self._last_close,
                "volume": 0.0,
            }
        self._current_second = new_second
        return finished


class KlineBarTracker:
    """Tracks OHLCV bars from Binance's native kline websocket stream
    (closed-candle events only, i.e. payload["k"]["x"] is True), gap-filling
    any silently-skipped interval with a flat bar at the last known close --
    the same gap-fill contract as BarAggregator.flush_to, just driven off
    Binance's own candle boundaries instead of raw trade timestamps, since
    Binance already does the OHLC aggregation for us here."""

    def __init__(self, bar_seconds: int):
        self.bar_seconds = bar_seconds
        self._lock = threading.Lock()
        self._last_second = None
        self._last_close = None

    def on_kline_closed(self, open_second: int, bar: dict):
        """Call once per closed kline event. Returns a list of (second, bar)
        tuples: gap-fill bar(s) for any whole interval(s) silently skipped
        since the last one received, followed by this real bar."""
        with self._lock:
            finished = []
            if self._last_second is not None:
                gap_second = self._last_second + self.bar_seconds
                while gap_second < open_second:
                    finished.append((gap_second, self._flat_bar()))
                    gap_second += self.bar_seconds
            finished.append((open_second, bar))
            self._last_second = open_second
            self._last_close = bar["close"]
            return finished

    def flush_to(self, wall_second: int):
        """Wall-clock watchdog: if the market has gone quiet long enough that
        Binance hasn't pushed a closed-kline event for a while, gap-fill up
        to (but not including) the currently-forming bar so downstream
        indicators don't stall indefinitely."""
        with self._lock:
            if self._last_second is None:
                return []
            current_bar_start = (wall_second // self.bar_seconds) * self.bar_seconds
            finished = []
            gap_second = self._last_second + self.bar_seconds
            while gap_second < current_bar_start:
                finished.append((gap_second, self._flat_bar()))
                gap_second += self.bar_seconds
            if finished:
                self._last_second = gap_second - self.bar_seconds
            return finished

    def _flat_bar(self):
        return {
            "open": self._last_close,
            "high": self._last_close,
            "low": self._last_close,
            "close": self._last_close,
            "volume": 0.0,
        }
