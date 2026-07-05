#!/usr/bin/env python3
"""Live HYPEUSDT trade stream -> 1s bars -> RSI/pivots -> hidden divergence
-> fine-tuned Qwen2.5-0.5B trade signal.

Uses the same rsid.indicators / rsid.divergence / rsid.prompt code paths as
the offline batch pipeline, so what the model sees live matches training.

Usage:
    python scripts/infer_stream.py
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.bars import BarAggregator
from rsid.divergence import StreamingDivergenceDetector
from rsid.indicators import IncrementalWilderRSI
from rsid.model import generate_signal, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--rsi-period", type=int, default=config.RSI_PERIOD)
    parser.add_argument("--pivot-lookback", type=int, default=config.PIVOT_LOOKBACK)
    parser.add_argument("--window", type=int, default=config.PROMPT_WINDOW_BARS)
    args = parser.parse_args()

    tokenizer, model = load_model()
    model.eval()

    rsi_calc = IncrementalWilderRSI(period=args.rsi_period)
    detector = StreamingDivergenceDetector(lookback=args.pivot_lookback)
    bar_window = deque(maxlen=args.window)
    aggregator = BarAggregator()
    processing_lock = threading.Lock()

    def process_bar(second: int, bar: dict):
        with processing_lock:
            ts = datetime.fromtimestamp(second, tz=timezone.utc)
            rsi = rsi_calc.update(bar["close"])
            bar_window.append({"timestamp": ts, "close": bar["close"], "rsi": rsi})
            event = detector.update(ts, bar["high"], bar["low"], bar["close"], rsi)

            rsi_str = f"{rsi:.2f}" if rsi is not None else "NA"
            print(f"{ts.isoformat()} close={bar['close']:.4f} rsi={rsi_str}")

            if event and len(bar_window) == bar_window.maxlen:
                print(f"  >>> divergence detected: {event['type']}")
                result = generate_signal(tokenizer, model, list(bar_window), event)
                print(f"  >>> model signal: {result}")

    def on_message(ws, message):
        data = json.loads(message)
        finished = aggregator.on_trade(float(data["p"]), float(data["q"]), int(data["T"]))
        for second, bar in finished:
            process_bar(second, bar)

    def on_error(ws, error):
        print(f"websocket error: {error}")

    def on_close(ws, code, msg):
        print(f"websocket closed: {code} {msg}")

    def ticker():
        while True:
            time.sleep(1)
            for second, bar in aggregator.flush_to(int(time.time())):
                process_bar(second, bar)

    stream_symbol = args.symbol.lower()
    url = f"{config.FUTURES_WS_BASE}/{stream_symbol}@aggTrade"
    print(f"connecting to {url}")
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)

    threading.Thread(target=ticker, daemon=True).start()
    ws.run_forever()


if __name__ == "__main__":
    main()
