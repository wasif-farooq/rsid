#!/usr/bin/env python3
"""Download Binance USD-M futures aggTrades daily dumps and resample to 1s OHLCV.

HYPEUSDT doesn't exist on Binance Spot and Binance Futures' REST kline API
doesn't support the 1s interval, so this reconstructs true 1-second candles
from tick-level trade data published at data.binance.vision.

Usage:
    python scripts/download_data.py --symbol HYPEUSDT --start 2025-05-30 --end 2025-06-05
"""

import argparse
import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

AGGTRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_day(symbol: str, day: date, market: str) -> bytes | None:
    url = f"{config.AGGTRADES_BASE_URL}/{market}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def parse_aggtrades(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            has_header = f.readline().decode().startswith("agg_trade_id")
        with zf.open(csv_name) as f:
            if has_header:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None, names=AGGTRADES_COLUMNS)
    return df


def resample_to_1s(trades: pd.DataFrame, day: date) -> pd.DataFrame:
    """Aggregate raw trades into 1s OHLCV bars, gap-filling seconds with no trades."""
    trades = trades.copy()
    trades["ts"] = pd.to_datetime(trades["transact_time"], unit="ms", utc=True)
    trades = trades.set_index("ts").sort_index()

    ohlc = trades["price"].resample("1s").ohlc()
    volume = trades["quantity"].resample("1s").sum()
    trade_count = trades["price"].resample("1s").count()

    day_start = pd.Timestamp(day, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    full_index = pd.date_range(day_start, day_end, freq="1s", tz="UTC")

    ohlc = ohlc.reindex(full_index)
    volume = volume.reindex(full_index, fill_value=0.0)
    trade_count = trade_count.reindex(full_index, fill_value=0)

    is_gap = ohlc["close"].isna()
    # No trade this second: flat bar at the last known close. Leading gaps
    # (no trade yet at all today) fall back to the day's first traded price.
    ohlc["close"] = ohlc["close"].ffill().bfill()
    for col in ("open", "high", "low"):
        ohlc[col] = ohlc[col].where(~is_gap, ohlc["close"])

    return pd.DataFrame(
        {
            "timestamp": full_index,
            "open": ohlc["open"].to_numpy(),
            "high": ohlc["high"].to_numpy(),
            "low": ohlc["low"].to_numpy(),
            "close": ohlc["close"].to_numpy(),
            "volume": volume.to_numpy(),
            "trade_count": trade_count.to_numpy(),
            "is_gap": is_gap.to_numpy(),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--start", default=config.EARLIEST_AVAILABLE_DATE)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out-dir", default=str(config.RAW_DIR))
    parser.add_argument("--market", default=config.MARKET)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    for day in daterange(start, end):
        out_path = out_dir / f"{args.symbol}_1s_{day.isoformat()}.parquet"
        if out_path.exists():
            print(f"skip {day} (already downloaded)")
            continue

        print(f"downloading {day}...", end=" ", flush=True)
        zip_bytes = None
        for attempt in range(2):
            try:
                zip_bytes = download_day(args.symbol, day, args.market)
                break
            except requests.RequestException as exc:
                print(f"error ({exc})", end=" ", flush=True)
                if attempt == 1:
                    print("giving up on this day")

        if zip_bytes is None:
            print("no data available, skipping")
            continue

        trades = parse_aggtrades(zip_bytes)
        bars = resample_to_1s(trades, day)
        bars.to_parquet(out_path, index=False)
        n_gap = int(bars["is_gap"].sum())
        print(f"saved {len(bars)} bars ({n_gap} gap-filled, {len(trades)} raw trades) -> {out_path.name}")


if __name__ == "__main__":
    main()
