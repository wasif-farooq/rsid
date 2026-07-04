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

import torch
import websocket
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.divergence import StreamingDivergenceDetector
from rsid.indicators import IncrementalWilderRSI
from rsid.prompt import build_messages, parse_completion


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


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.LORA_MERGED_DIR.exists() and any(config.LORA_MERGED_DIR.iterdir()):
        print(f"loading merged fine-tuned model from {config.LORA_MERGED_DIR}")
        path = str(config.LORA_MERGED_DIR)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
        return tokenizer, model

    if config.LORA_ADAPTER_DIR.exists() and any(config.LORA_ADAPTER_DIR.iterdir()):
        from peft import PeftModel

        print(f"loading base model + LoRA adapter from {config.LORA_ADAPTER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(config.LORA_ADAPTER_DIR))
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.bfloat16, device_map=device)
        model = PeftModel.from_pretrained(base, str(config.LORA_ADAPTER_DIR))
        return tokenizer, model

    print("no fine-tuned model found -- falling back to base model (untrained on this task)")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.bfloat16, device_map=device)
    return tokenizer, model


def generate_signal(tokenizer, model, bars: list[dict], event: dict) -> dict:
    messages = build_messages(bars, event)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_completion(text)


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
