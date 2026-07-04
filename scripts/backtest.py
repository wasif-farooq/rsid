#!/usr/bin/env python3
"""Backtest the fine-tuned model as a mechanical trading strategy.

Regenerates the same train/val split scripts/build_dataset.py produces
(same seed, same hold-cap/window/negative-ratio defaults), but keeps the
iloc/outcome metadata build_dataset.py discards before writing JSONL. For
every validation example the model is run to get a signal; BUY/SELL
predictions are walked forward through price using the same TP/SL/lookahead
rules as scripts/label_divergence.py to get a realized return, and a
"trade every real divergence event" baseline is computed the same way over
the same sample for comparison.

Usage:
    python scripts/backtest.py --sample-size 500
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.divergence import HIDDEN_BEARISH, HIDDEN_BULLISH
from rsid.paths import find_latest
from rsid.prompt import SIGNAL_HOLD, build_messages, parse_completion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import make_bars_window, sample_negative_ilocs


def simulate_trade(close, high, low, anchor_iloc, direction, lookahead_bars, tp_pct, sl_pct):
    """Like label_divergence.label_outcome, but also returns a realized pct
    return for every path (tp/sl/timeout mark-to-market), so PnL can be
    aggregated regardless of how the trade resolved."""
    anchor_price = close[anchor_iloc]
    n = len(close)
    end = min(anchor_iloc + lookahead_bars, n - 1)
    sign = 1 if direction == HIDDEN_BULLISH else -1

    if direction == HIDDEN_BULLISH:
        tp_price = anchor_price * (1 + tp_pct)
        sl_price = anchor_price * (1 - sl_pct)
    else:
        tp_price = anchor_price * (1 - tp_pct)
        sl_price = anchor_price * (1 + sl_pct)

    for i in range(anchor_iloc + 1, end + 1):
        if direction == HIDDEN_BULLISH:
            hit_sl = low[i] <= sl_price
            hit_tp = high[i] >= tp_price
        else:
            hit_sl = high[i] >= sl_price
            hit_tp = low[i] <= tp_price
        if hit_sl:
            return {"outcome": "sl", "pct_return": -sl_pct}
        if hit_tp:
            return {"outcome": "tp", "pct_return": tp_pct}

    mtm_pct = sign * (close[end] / anchor_price - 1)
    return {"outcome": "timeout", "pct_return": mtm_pct}


def build_examples_with_metadata(df, labeled, window, negative_ratio, hold_cap_ratio, seed):
    """Mirrors scripts/build_dataset.py's example construction exactly, but
    keeps anchor_iloc/type/outcome alongside each example instead of
    discarding it, so the backtest can walk price forward per example."""
    rng = random.Random(seed)

    if hold_cap_ratio > 0:
        is_hold = labeled["signal_label"] == SIGNAL_HOLD
        n_directional = int((~is_hold).sum())
        hold_cap = int(n_directional * hold_cap_ratio)
        if int(is_hold.sum()) > hold_cap:
            hold_kept = labeled[is_hold].sample(n=hold_cap, random_state=seed)
            labeled = pd.concat([labeled[~is_hold], hold_kept]).sort_values("confirmed_iloc").reset_index(drop=True)

    examples = []
    event_ilocs = set()
    for row in labeled.itertuples(index=False):
        end_iloc = int(row.confirmed_iloc)
        event_ilocs.add(end_iloc)
        bars = make_bars_window(df, end_iloc, window)
        if bars is None:
            continue
        event = {
            "type": row.type,
            "rsi_value": row.rsi_value,
            "prev_rsi_value": row.prev_rsi_value,
            "price_value": row.price_value,
            "prev_price_value": row.prev_price_value,
        }
        examples.append(
            {
                "anchor_timestamp": row.confirmed_timestamp,
                "anchor_iloc": end_iloc,
                "bars": bars,
                "event": event,
                "true_type": row.type,
                "true_signal": row.signal_label,
                "is_real_event": True,
            }
        )

    n_negative = int(len(examples) * negative_ratio)
    if n_negative > 0:
        neg_ilocs = sample_negative_ilocs(df, set(event_ilocs), window, n_negative, rng)
        for i in neg_ilocs:
            bars = make_bars_window(df, i, window)
            if bars is None:
                continue
            examples.append(
                {
                    "anchor_timestamp": df["timestamp"].iloc[i],
                    "anchor_iloc": i,
                    "bars": bars,
                    "event": None,
                    "true_type": None,
                    "true_signal": SIGNAL_HOLD,
                    "is_real_event": False,
                }
            )

    examples.sort(key=lambda e: e["anchor_timestamp"])
    return examples


def load_model():
    path = str(config.LORA_MERGED_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return tokenizer, model


def generate_signal(tokenizer, model, bars, event):
    messages = build_messages(bars, event)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_completion(text)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--labeled-path", default=None)
    parser.add_argument("--window", type=int, default=config.PROMPT_WINDOW_BARS)
    parser.add_argument("--negative-ratio", type=float, default=config.NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--hold-cap-ratio", type=float, default=config.HOLD_CAP_RATIO)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_SPLIT_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=500, help="random sample drawn from the val split")
    parser.add_argument("--lookahead-seconds", type=int, default=config.LOOKAHEAD_SECONDS)
    parser.add_argument("--take-profit-pct", type=float, default=config.TAKE_PROFIT_PCT)
    parser.add_argument("--stop-loss-pct", type=float, default=config.STOP_LOSS_PCT)
    args = parser.parse_args()

    processed_dir = config.PROCESSED_DIR
    features_path = Path(args.features_path) if args.features_path else find_latest(processed_dir, f"{config.SYMBOL}_features_*.parquet")
    labeled_path = Path(args.labeled_path) if args.labeled_path else find_latest(processed_dir, f"{config.SYMBOL}_*_labeled.parquet")
    print(f"features: {features_path}")
    print(f"labeled:  {labeled_path}")

    df = pd.read_parquet(features_path)
    labeled = pd.read_parquet(labeled_path)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    print("reconstructing build_dataset.py's example list (with iloc metadata kept)...")
    examples = build_examples_with_metadata(df, labeled, args.window, args.negative_ratio, args.hold_cap_ratio, args.seed)
    n_val = max(1, int(len(examples) * args.val_fraction))
    train_examples = examples[: len(examples) - n_val]
    val_examples = examples[len(examples) - n_val :]
    print(f"reconstructed: train={len(train_examples)} val={len(val_examples)} (build_dataset.py produced train=86307 val=9589)")

    rng = random.Random(args.seed)
    sample = val_examples if len(val_examples) <= args.sample_size else rng.sample(val_examples, args.sample_size)
    print(f"backtesting on {len(sample)} sampled val examples...")

    tokenizer, model = load_model()

    lookahead_bars = args.lookahead_seconds  # 1 bar == 1 second
    model_trades = []
    baseline_trades = []
    confusion = {}
    t0 = time.monotonic()
    for idx, ex in enumerate(sample):
        pred = generate_signal(tokenizer, model, ex["bars"], ex["event"])
        key = (ex["true_signal"], pred["signal"])
        confusion[key] = confusion.get(key, 0) + 1

        if pred["signal"] in ("BUY", "SELL"):
            direction = HIDDEN_BULLISH if pred["signal"] == "BUY" else HIDDEN_BEARISH
            result = simulate_trade(close, high, low, ex["anchor_iloc"], direction, lookahead_bars, args.take_profit_pct, args.stop_loss_pct)
            model_trades.append(result)

        if ex["is_real_event"]:
            true_direction = HIDDEN_BULLISH if ex["true_type"] == HIDDEN_BULLISH else HIDDEN_BEARISH
            baseline_result = simulate_trade(close, high, low, ex["anchor_iloc"], true_direction, lookahead_bars, args.take_profit_pct, args.stop_loss_pct)
            baseline_trades.append(baseline_result)

        if (idx + 1) % 50 == 0 or idx + 1 == len(sample):
            elapsed = time.monotonic() - t0
            rate = (idx + 1) / elapsed
            eta = (len(sample) - idx - 1) / rate
            print(f"  [{idx + 1}/{len(sample)}] {rate:.2f} ex/s, elapsed {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    print("\n=== confusion (true_signal, predicted_signal) -> count ===")
    for k, v in sorted(confusion.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    print("\n=== backtest results ===")
    summarize(model_trades, "model-driven strategy (trade every model BUY/SELL)")
    summarize(baseline_trades, "baseline: trade every real divergence event long/short by true type")


if __name__ == "__main__":
    main()
