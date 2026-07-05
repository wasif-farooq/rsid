#!/usr/bin/env python3
"""Summarize live paper-trading results from trades.jsonl using the exact
same aggregation as scripts/backtest.py, so live and historical numbers are
never produced by two hand-maintained copies of the same math.

Usage:
    python scripts/summarize_paper_trades.py
    python scripts/summarize_paper_trades.py --trade-log-path data/paper_trading/trades.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.reporting import summarize


def load_closed_trades(path: Path) -> list[dict]:
    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "trade_close":
                trades.append(record)
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-log-path", default=str(config.PAPER_TRADE_LOG_PATH))
    args = parser.parse_args()

    path = Path(args.trade_log_path)
    if not path.exists():
        print(f"no trade log found at {path}")
        return

    trades = load_closed_trades(path)
    print(f"loaded {len(trades)} closed trades from {path}")
    summarize(trades, "paper trading")


if __name__ == "__main__":
    main()
