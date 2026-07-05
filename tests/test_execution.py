"""PaperTradingEngine must resolve TP/SL/timeout identically to
scripts/backtest.py::simulate_trade -- these tests feed the same synthetic
high/low/close arrays into both the incremental engine (bar-by-bar) and the
array-slice reference implementation and assert identical outcome/pct_return,
including a same-bar TP+SL tie (stop-loss must win both ways)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backtest import simulate_trade
from rsid.divergence import HIDDEN_BEARISH, HIDDEN_BULLISH
from rsid.execution import PaperTradingEngine

TP_PCT = 0.02
SL_PCT = 0.01
LOOKAHEAD_BARS = 10


def _run_reference(close, high, low, direction):
    return simulate_trade(
        np.array(close, dtype=float),
        np.array(high, dtype=float),
        np.array(low, dtype=float),
        anchor_iloc=0,
        direction=direction,
        lookahead_bars=LOOKAHEAD_BARS,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
    )


def _run_engine(close, high, low, direction):
    signal = "BUY" if direction == HIDDEN_BULLISH else "SELL"
    engine = PaperTradingEngine(notional_usd=100.0, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead_bars=LOOKAHEAD_BARS)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine.on_signal(signal, t0, close[0], event=None)

    trade = None
    for i in range(1, len(close)):
        ts = t0 + timedelta(seconds=i)
        result = engine.on_bar(ts, high[i], low[i], close[i])
        if result is not None:
            trade = result
            break
    return trade


def test_take_profit_hit_bullish():
    close = [100.0] + [100.0] * 9
    high = [100.0, 100.5, 101.0, 102.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    low = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

    ref = _run_reference(close, high, low, HIDDEN_BULLISH)
    got = _run_engine(close, high, low, HIDDEN_BULLISH)

    assert ref["outcome"] == "tp"
    assert got["outcome"] == ref["outcome"]
    assert got["pct_return"] == ref["pct_return"]


def test_stop_loss_hit_bearish():
    close = [100.0] + [100.0] * 9
    high = [100.0, 100.0, 100.0, 101.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    low = [100.0] * 10

    ref = _run_reference(close, high, low, HIDDEN_BEARISH)
    got = _run_engine(close, high, low, HIDDEN_BEARISH)

    assert ref["outcome"] == "sl"
    assert got["outcome"] == ref["outcome"]
    assert got["pct_return"] == ref["pct_return"]


def test_timeout_marks_to_market():
    close = [100.0, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 100.95, 101.0]
    high = [c + 0.05 for c in close]
    low = [c - 0.05 for c in close]

    ref = _run_reference(close, high, low, HIDDEN_BULLISH)
    got = _run_engine(close, high, low, HIDDEN_BULLISH)

    assert ref["outcome"] == "timeout"
    assert got["outcome"] == ref["outcome"]
    assert abs(got["pct_return"] - ref["pct_return"]) < 1e-12


def test_same_bar_tp_and_sl_tie_sl_wins_bullish():
    # Bar 1 touches both tp_price (102.0) and sl_price (99.0) -- SL must win.
    close = [100.0] + [100.0] * 9
    high = [100.0, 103.0] + [100.0] * 8
    low = [100.0, 98.0] + [100.0] * 8

    ref = _run_reference(close, high, low, HIDDEN_BULLISH)
    got = _run_engine(close, high, low, HIDDEN_BULLISH)

    assert ref["outcome"] == "sl"
    assert got["outcome"] == "sl"
    assert got["pct_return"] == ref["pct_return"] == -SL_PCT


def test_same_bar_tp_and_sl_tie_sl_wins_bearish():
    close = [100.0] + [100.0] * 9
    high = [100.0, 102.0] + [100.0] * 8
    low = [100.0, 97.0] + [100.0] * 8

    ref = _run_reference(close, high, low, HIDDEN_BEARISH)
    got = _run_engine(close, high, low, HIDDEN_BEARISH)

    assert ref["outcome"] == "sl"
    assert got["outcome"] == "sl"
    assert got["pct_return"] == ref["pct_return"] == -SL_PCT


def test_new_signal_ignored_while_position_open():
    engine = PaperTradingEngine(notional_usd=100.0, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead_bars=LOOKAHEAD_BARS)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    opened = engine.on_signal("BUY", t0, 100.0, event=None)
    assert opened["action"] == "open"

    ignored = engine.on_signal("SELL", t0 + timedelta(seconds=1), 100.0, event=None)
    assert ignored["action"] == "ignored_open_position"
    assert engine.position.signal == "BUY"


def test_state_round_trip_resumes_open_position():
    engine = PaperTradingEngine(notional_usd=100.0, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead_bars=LOOKAHEAD_BARS)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine.on_signal("BUY", t0, 100.0, event=None)

    state = engine.to_state_dict()
    restored = PaperTradingEngine.from_state_dict(
        state, notional_usd=100.0, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead_bars=LOOKAHEAD_BARS
    )

    assert restored.position is not None
    assert restored.position.entry_price == 100.0
    assert restored.position.signal == "BUY"
    assert restored.position.bars_since_entry == 0
