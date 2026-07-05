"""Streaming (one-bar-at-a-time) TP/SL paper-trading engine.

Mirrors scripts/backtest.py::simulate_trade / scripts/label_divergence.py::
label_outcome exactly, but incrementally: instead of slicing a finished
close/high/low array from a known anchor_iloc, a position accumulates
bars_since_entry as new bars arrive live, and resolves tp/sl/timeout using
the same tie-break (stop-loss wins a same-bar tp+sl touch, the conservative
assumption both those functions already make) and the same mark-to-market
formula at timeout. This equivalence is what keeps live paper-trading
results comparable to the historical backtest numbers.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rsid.divergence import HIDDEN_BEARISH, HIDDEN_BULLISH


@dataclass
class Position:
    direction: str  # HIDDEN_BULLISH / HIDDEN_BEARISH
    signal: str  # "BUY" / "SELL"
    entry_timestamp: Any
    entry_price: float
    notional_usd: float
    qty: float  # notional_usd / entry_price
    tp_price: float
    sl_price: float
    bars_since_entry: int = 0
    entry_event: dict | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        ts = d["entry_timestamp"]
        d["entry_timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else ts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        ts = d["entry_timestamp"]
        d["entry_timestamp"] = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        return cls(**d)


class PaperTradingEngine:
    """One-position-at-a-time TP/SL simulator. New BUY/SELL signals while a
    position is open are ignored (log-only) rather than queued or stacked."""

    def __init__(self, notional_usd: float, tp_pct: float, sl_pct: float, lookahead_bars: int):
        self.notional_usd = notional_usd
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.lookahead_bars = lookahead_bars
        self._position: Position | None = None

    @property
    def position(self) -> Position | None:
        return self._position

    def on_signal(self, signal: str, timestamp, price: float, event: dict | None) -> dict:
        """BUY/SELL with no open position opens one anchored at `price` (this
        bar's close); TP/SL checks for it begin on the *next* bar, mirroring
        simulate_trade's loop starting at anchor_iloc + 1. BUY/SELL while a
        position is already open is ignored. HOLD is always a no-op."""
        if signal not in ("BUY", "SELL"):
            return {"action": "none", "signal": signal, "timestamp": timestamp}

        if self._position is not None:
            return {
                "action": "ignored_open_position",
                "signal": signal,
                "timestamp": timestamp,
                "reason": "position already open",
            }

        if price is None or price != price or price <= 0:  # None/NaN/non-positive guard
            return {
                "action": "rejected",
                "signal": signal,
                "timestamp": timestamp,
                "reason": f"invalid entry price {price!r}",
            }

        direction = HIDDEN_BULLISH if signal == "BUY" else HIDDEN_BEARISH
        if direction == HIDDEN_BULLISH:
            tp_price = price * (1 + self.tp_pct)
            sl_price = price * (1 - self.sl_pct)
        else:
            tp_price = price * (1 - self.tp_pct)
            sl_price = price * (1 + self.sl_pct)

        self._position = Position(
            direction=direction,
            signal=signal,
            entry_timestamp=timestamp,
            entry_price=price,
            notional_usd=self.notional_usd,
            qty=self.notional_usd / price,
            tp_price=tp_price,
            sl_price=sl_price,
            bars_since_entry=0,
            entry_event=event,
        )
        return {
            "action": "open",
            "signal": signal,
            "timestamp": timestamp,
            "entry_price": price,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

    def on_bar(self, timestamp, high: float, low: float, close: float) -> dict | None:
        """No open position -> None. Open position -> bars_since_entry += 1,
        check SL then TP (SL wins a same-bar tie) then timeout-at-
        lookahead_bars (mark-to-market against this bar's close); if
        resolved, clears the position and returns the closed-trade record."""
        pos = self._position
        if pos is None:
            return None

        pos.bars_since_entry += 1
        sign = 1 if pos.direction == HIDDEN_BULLISH else -1

        if pos.direction == HIDDEN_BULLISH:
            hit_sl = low <= pos.sl_price
            hit_tp = high >= pos.tp_price
        else:
            hit_sl = high >= pos.sl_price
            hit_tp = low <= pos.tp_price

        if hit_sl:
            outcome, pct_return = "sl", -self.sl_pct
        elif hit_tp:
            outcome, pct_return = "tp", self.tp_pct
        elif pos.bars_since_entry >= self.lookahead_bars:
            outcome, pct_return = "timeout", sign * (close / pos.entry_price - 1)
        else:
            return None

        trade = {
            "signal": pos.signal,
            "direction": pos.direction,
            "entry_timestamp": pos.entry_timestamp,
            "entry_price": pos.entry_price,
            "exit_timestamp": timestamp,
            "exit_price": close,
            "notional_usd": pos.notional_usd,
            "qty": pos.qty,
            "bars_held": pos.bars_since_entry,
            "outcome": outcome,
            "pct_return": pct_return,
            "pnl_usd": pos.notional_usd * pct_return,
            "entry_event": pos.entry_event,
        }
        self._position = None
        return trade

    def to_state_dict(self) -> dict:
        return {"position": self._position.to_dict() if self._position else None}

    @classmethod
    def from_state_dict(
        cls, state: dict | None, notional_usd: float, tp_pct: float, sl_pct: float, lookahead_bars: int
    ) -> "PaperTradingEngine":
        engine = cls(notional_usd, tp_pct, sl_pct, lookahead_bars)
        pos_dict = state.get("position") if state else None
        if pos_dict:
            engine._position = Position.from_dict(pos_dict)
        return engine
