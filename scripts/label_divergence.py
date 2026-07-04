#!/usr/bin/env python3
"""Detect hidden RSI divergence events and (in batch mode) label them with
a forward-return trade outcome.

--mode batch: runs the divergence detector over a features parquet, then
for every event walks forward up to --lookahead-seconds to see whether
take-profit or stop-loss is hit first, producing signal_label in
{BUY, SELL, HOLD}. This is what scripts/build_dataset.py consumes.

--mode stream: replays the same features parquet bar-by-bar through the
causal StreamingDivergenceDetector (no forward peeking, no signal_label).
This exists to validate that streaming and batch detection agree, and
documents exactly what scripts/infer_stream.py sees live.

Usage:
    python scripts/label_divergence.py --mode batch
    # or explicitly:
    python scripts/label_divergence.py --mode batch \
        --features-path data/processed/HYPEUSDT_features_2025-05-30_2025-05-30.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.divergence import (
    HIDDEN_BEARISH,
    HIDDEN_BULLISH,
    StreamingDivergenceDetector,
    detect_hidden_divergence_batch,
)
from rsid.paths import find_latest
from rsid.prompt import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL


def label_outcome(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    anchor_iloc: int,
    direction: str,
    lookahead_bars: int,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> dict:
    """Walk forward from anchor_iloc (1 bar per second) to see whether TP or
    SL resolves first. direction is HIDDEN_BULLISH (long) or HIDDEN_BEARISH
    (short). If both TP and SL are touched within the same bar, SL is
    assumed to have hit first (conservative, standard backtesting practice).
    """
    anchor_price = close[anchor_iloc]
    n = len(close)
    end = min(anchor_iloc + lookahead_bars, n - 1)

    if direction == HIDDEN_BULLISH:
        tp_price = anchor_price * (1 + take_profit_pct)
        sl_price = anchor_price * (1 - stop_loss_pct)
        signal, opposite = SIGNAL_BUY, SIGNAL_HOLD
    else:
        tp_price = anchor_price * (1 - take_profit_pct)
        sl_price = anchor_price * (1 + stop_loss_pct)
        signal, opposite = SIGNAL_SELL, SIGNAL_HOLD

    for i in range(anchor_iloc + 1, end + 1):
        if direction == HIDDEN_BULLISH:
            hit_sl = low[i] <= sl_price
            hit_tp = high[i] >= tp_price
        else:
            hit_sl = high[i] >= sl_price
            hit_tp = low[i] <= tp_price

        if hit_sl:
            return {"signal_label": opposite, "resolved_bars": i - anchor_iloc, "outcome": "sl"}
        if hit_tp:
            return {"signal_label": signal, "resolved_bars": i - anchor_iloc, "outcome": "tp"}

    return {"signal_label": opposite, "resolved_bars": None, "outcome": "timeout"}


def run_batch(df: pd.DataFrame, lookback: int, lookahead_seconds: int, tp_pct: float, sl_pct: float):
    print(f"detecting divergence pivots over {len(df)} bars...")
    t0 = time.monotonic()
    events = detect_hidden_divergence_batch(df, lookback=lookback)
    print(f"found {len(events)} divergence events in {time.monotonic() - t0:.1f}s; labeling outcomes...")

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy()

    rows = []
    t0 = time.monotonic()
    last_print = t0
    n = len(events)
    for idx, e in enumerate(events):
        outcome = label_outcome(close, high, low, e["confirmed_iloc"], e["type"], lookahead_seconds, tp_pct, sl_pct)
        rows.append(
            {
                "confirmed_iloc": e["confirmed_iloc"],
                "confirmed_timestamp": timestamps[e["confirmed_iloc"]],
                "pivot_iloc": e["pivot_iloc"],
                "pivot_timestamp": timestamps[e["pivot_iloc"]],
                "type": e["type"],
                "rsi_value": e["rsi_value"],
                "prev_rsi_value": e["prev_rsi_value"],
                "price_value": e["price_value"],
                "prev_price_value": e["prev_price_value"],
                **outcome,
            }
        )

        now = time.monotonic()
        if now - last_print >= 2.0 or idx + 1 == n:
            elapsed = now - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx + 1}/{n}] {rate:.1f} events/s, "
                f"elapsed {elapsed:.0f}s, ETA {remaining:.0f}s",
                flush=True,
            )
            last_print = now

    return pd.DataFrame(rows)


def run_stream_replay(df: pd.DataFrame, lookback: int):
    """Causal replay for validation; returns events with no outcome label."""
    print(f"replaying {len(df)} bars through the streaming detector...")
    detector = StreamingDivergenceDetector(lookback=lookback)
    rows = []
    n = len(df)
    t0 = time.monotonic()
    last_print = t0
    for idx, row in enumerate(df.itertuples(index=False)):
        rsi = None if pd.isna(row.rsi) else row.rsi
        event = detector.update(row.timestamp, row.high, row.low, row.close, rsi)
        if event:
            rows.append(event)

        now = time.monotonic()
        if now - last_print >= 2.0 or idx + 1 == n:
            elapsed = now - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx + 1}/{n}] {rate:.0f} bars/s, elapsed {elapsed:.0f}s, ETA {remaining:.0f}s", flush=True)
            last_print = now

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["batch", "stream"], default="batch")
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument(
        "--features-path",
        default=None,
        help="Defaults to the most recently modified data/processed/{symbol}_features_*.parquet",
    )
    parser.add_argument("--out-dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--pivot-lookback", type=int, default=config.PIVOT_LOOKBACK)
    parser.add_argument("--lookahead-seconds", type=int, default=config.LOOKAHEAD_SECONDS)
    parser.add_argument("--take-profit-pct", type=float, default=config.TAKE_PROFIT_PCT)
    parser.add_argument("--stop-loss-pct", type=float, default=config.STOP_LOSS_PCT)
    args = parser.parse_args()

    if args.features_path:
        features_path = Path(args.features_path)
    else:
        features_path = find_latest(Path(args.out_dir), f"{args.symbol}_features_*.parquet")
        print(f"--features-path not given, using most recent: {features_path}")

    print(f"loading {features_path} ({features_path.stat().st_size / 1e6:.0f} MB)...")
    t_load = time.monotonic()
    df = pd.read_parquet(features_path)
    print(f"loaded {len(df)} rows in {time.monotonic() - t_load:.1f}s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = features_path.stem.replace("_features_", "_")

    if args.mode == "batch":
        labeled = run_batch(df, args.pivot_lookback, args.lookahead_seconds, args.take_profit_pct, args.stop_loss_pct)
        out_path = out_dir / f"{stem}_labeled.parquet"
        labeled.to_parquet(out_path, index=False)
        print(f"events found: {len(labeled)}")
        if len(labeled):
            print(labeled["type"].value_counts())
            print(labeled["signal_label"].value_counts())
        print(f"saved -> {out_path}")
    else:
        events = run_stream_replay(df, args.pivot_lookback)
        out_path = out_dir / f"{stem}_stream_events.parquet"
        events.to_parquet(out_path, index=False)
        print(f"stream events found: {len(events)}")
        if len(events):
            print(events["type"].value_counts())
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
