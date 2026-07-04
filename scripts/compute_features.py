#!/usr/bin/env python3
"""Load raw 1s OHLCV parquets, compute RSI(14) and HH/HL/LH/LL swing-pivot
labels on both price and RSI, and write a consolidated feature parquet.

Usage:
    python scripts/compute_features.py --start 2025-05-30 --end 2025-06-05
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.indicators import find_pivots, label_pivot_events, wilder_rsi


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_raw(symbol: str, start: date, end: date, raw_dir: Path) -> pd.DataFrame:
    frames = []
    for day in daterange(start, end):
        path = raw_dir / f"{symbol}_1s_{day.isoformat()}.parquet"
        if not path.exists():
            print(f"missing {path.name}, skipping")
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no raw parquet files found for {symbol} in [{start}, {end}]")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _assign_pivot_columns(df: pd.DataFrame, prefix: str, highs: list[dict], lows: list[dict]):
    df[f"{prefix}_high_pivot"] = False
    df[f"{prefix}_high_label"] = None
    df[f"{prefix}_low_pivot"] = False
    df[f"{prefix}_low_label"] = None

    high_pivot_col = df.columns.get_loc(f"{prefix}_high_pivot")
    high_label_col = df.columns.get_loc(f"{prefix}_high_label")
    low_pivot_col = df.columns.get_loc(f"{prefix}_low_pivot")
    low_label_col = df.columns.get_loc(f"{prefix}_low_label")

    for e in highs:
        df.iat[e["pivot_iloc"], high_pivot_col] = True
        df.iat[e["pivot_iloc"], high_label_col] = e["label"]
    for e in lows:
        df.iat[e["pivot_iloc"], low_pivot_col] = True
        df.iat[e["pivot_iloc"], low_label_col] = e["label"]


def add_features(df: pd.DataFrame, rsi_period: int, pivot_lookback: int) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = wilder_rsi(df["close"], period=rsi_period)

    # Price swing highs come from the `high` series, swing lows from `low` --
    # standard TA convention. RSI swing highs/lows both come from the single
    # RSI line. These are informational/inspectable columns; the actual
    # hidden-divergence rule (rsid/divergence.py) recomputes RSI pivots
    # itself using these same shared functions.
    price_highs, _ = find_pivots(df["high"], pivot_lookback)
    _, price_lows = find_pivots(df["low"], pivot_lookback)
    price_highs = label_pivot_events(price_highs, kind="high")
    price_lows = label_pivot_events(price_lows, kind="low")
    _assign_pivot_columns(df, "price", price_highs, price_lows)

    rsi_highs, rsi_lows = find_pivots(df["rsi"], pivot_lookback)
    rsi_highs = label_pivot_events(rsi_highs, kind="high")
    rsi_lows = label_pivot_events(rsi_lows, kind="low")
    _assign_pivot_columns(df, "rsi", rsi_highs, rsi_lows)

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--start", default=config.EARLIEST_AVAILABLE_DATE)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--raw-dir", default=str(config.RAW_DIR))
    parser.add_argument("--out-dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--rsi-period", type=int, default=config.RSI_PERIOD)
    parser.add_argument("--pivot-lookback", type=int, default=config.PIVOT_LOOKBACK)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading raw data for {args.symbol} [{start} .. {end}]...")
    df = load_raw(args.symbol, start, end, Path(args.raw_dir))
    print(f"loaded {len(df)} bars")

    print("computing RSI and pivots...")
    df = add_features(df, args.rsi_period, args.pivot_lookback)

    out_path = out_dir / f"{args.symbol}_features_{start.isoformat()}_{end.isoformat()}.parquet"
    df.to_parquet(out_path, index=False)

    n_price_high = int(df["price_high_pivot"].sum())
    n_price_low = int(df["price_low_pivot"].sum())
    n_rsi_high = int(df["rsi_high_pivot"].sum())
    n_rsi_low = int(df["rsi_low_pivot"].sum())
    print(
        f"pivots: price_high={n_price_high} price_low={n_price_low} "
        f"rsi_high={n_rsi_high} rsi_low={n_rsi_low}"
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
