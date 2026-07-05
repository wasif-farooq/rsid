"""Shared trade-aggregation/reporting used by both the offline backtest
(scripts/backtest.py) and live paper-trading (scripts/summarize_paper_trades.py),
so historical and live performance numbers are always produced by the same code."""


def summarize(trades, label):
    n = len(trades)
    if n == 0:
        print(f"{label}: no trades taken")
        return
    wins = sum(1 for t in trades if t["pct_return"] > 0)
    total = sum(t["pct_return"] for t in trades)
    avg = total / n
    tp = sum(1 for t in trades if t["outcome"] == "tp")
    sl = sum(1 for t in trades if t["outcome"] == "sl")
    timeout = sum(1 for t in trades if t["outcome"] == "timeout")
    print(
        f"{label}: {n} trades | win rate {wins / n:.1%} | avg return/trade {avg:+.3%} | "
        f"total return (sum, no compounding) {total:+.2%} | tp={tp} sl={sl} timeout={timeout}"
    )
