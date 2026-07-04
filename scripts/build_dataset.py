#!/usr/bin/env python3
"""Turn labeled hidden-divergence events (+ a sample of clean no-divergence
bars) into a JSONL chat-format SFT dataset for fine-tuning Qwen2.5-0.5B.

Usage:
    python scripts/build_dataset.py
    # or explicitly:
    python scripts/build_dataset.py \
        --features-path data/processed/HYPEUSDT_features_2025-05-30_2025-05-30.parquet \
        --labeled-path data/processed/HYPEUSDT_2025-05-30_2025-05-30_labeled.parquet
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.paths import find_latest
from rsid.prompt import SIGNAL_HOLD, build_completion, build_messages


def make_bars_window(df: pd.DataFrame, end_iloc: int, window: int) -> list[dict] | None:
    """Trailing window of bars ending at (and including) end_iloc. Never
    reaches past end_iloc -- that's the causal boundary for this example.
    """
    start_iloc = end_iloc - window + 1
    if start_iloc < 0:
        return None
    sub = df.iloc[start_iloc : end_iloc + 1]
    if sub["rsi"].isna().any():
        return None
    return [{"timestamp": r.timestamp, "close": r.close, "rsi": r.rsi} for r in sub.itertuples(index=False)]


def build_example(bars: list[dict], event: dict | None, divergence_type, signal: str, anchor_timestamp) -> dict:
    messages = build_messages(bars, event)
    messages.append({"role": "assistant", "content": build_completion(divergence_type, signal)})
    return {"messages": messages, "_anchor_timestamp": anchor_timestamp}


def sample_negative_ilocs(df: pd.DataFrame, exclude_ilocs: set[int], window: int, n: int, rng: random.Random) -> list[int]:
    """Pick bar positions far from any divergence event/window, for HOLD
    'no divergence' training examples."""
    n_rows = len(df)
    candidates = []
    lo = window - 1
    hi = n_rows - 1
    if hi <= lo:
        return []
    attempts = 0
    max_attempts = n * 50
    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        i = rng.randint(lo, hi)
        if i in exclude_ilocs:
            continue
        candidates.append(i)
        exclude_ilocs.add(i)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument(
        "--features-path",
        default=None,
        help="Defaults to the most recently modified data/processed/{symbol}_features_*.parquet",
    )
    parser.add_argument(
        "--labeled-path",
        default=None,
        help="Defaults to the most recently modified data/processed/{symbol}_*_labeled.parquet",
    )
    parser.add_argument("--processed-dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--out-dir", default=str(config.DATASET_DIR))
    parser.add_argument("--window", type=int, default=config.PROMPT_WINDOW_BARS)
    parser.add_argument("--negative-ratio", type=float, default=config.NEGATIVE_SAMPLE_RATIO)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_SPLIT_FRACTION)
    parser.add_argument(
        "--hold-cap-ratio",
        type=float,
        default=config.HOLD_CAP_RATIO,
        help="Cap HOLD-labeled divergence events at this multiple of the BUY+SELL count "
        "(0 or negative disables capping).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if args.features_path:
        features_path = Path(args.features_path)
    else:
        features_path = find_latest(processed_dir, f"{args.symbol}_features_*.parquet")
        print(f"--features-path not given, using most recent: {features_path}")

    if args.labeled_path:
        labeled_path = Path(args.labeled_path)
    else:
        labeled_path = find_latest(processed_dir, f"{args.symbol}_*_labeled.parquet")
        print(f"--labeled-path not given, using most recent: {labeled_path}")

    rng = random.Random(args.seed)
    df = pd.read_parquet(features_path)
    labeled = pd.read_parquet(labeled_path)

    if args.hold_cap_ratio > 0:
        is_hold = labeled["signal_label"] == SIGNAL_HOLD
        n_directional = int((~is_hold).sum())
        hold_cap = int(n_directional * args.hold_cap_ratio)
        n_hold = int(is_hold.sum())
        if n_hold > hold_cap:
            hold_kept = labeled[is_hold].sample(n=hold_cap, random_state=args.seed)
            labeled = pd.concat([labeled[~is_hold], hold_kept]).sort_values("confirmed_iloc").reset_index(drop=True)
            print(
                f"downsampled HOLD divergence events {n_hold} -> {hold_cap} "
                f"({args.hold_cap_ratio}x the {n_directional} BUY+SELL events)"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    event_ilocs = set()
    skipped_short_history = 0

    print(f"building {len(labeled)} divergence-event prompts...")
    t0 = time.monotonic()
    last_print = t0
    n_labeled = len(labeled)
    for idx, row in enumerate(labeled.itertuples(index=False)):
        end_iloc = int(row.confirmed_iloc)
        event_ilocs.add(end_iloc)
        bars = make_bars_window(df, end_iloc, args.window)
        if bars is None:
            skipped_short_history += 1
            continue
        event = {
            "type": row.type,
            "rsi_value": row.rsi_value,
            "prev_rsi_value": row.prev_rsi_value,
            "price_value": row.price_value,
            "prev_price_value": row.prev_price_value,
        }
        examples.append(build_example(bars, event, row.type, row.signal_label, row.confirmed_timestamp))

        now = time.monotonic()
        if now - last_print >= 2.0 or idx + 1 == n_labeled:
            print(f"  [{idx + 1}/{n_labeled}] elapsed {now - t0:.0f}s", flush=True)
            last_print = now

    n_negative = int(len(examples) * args.negative_ratio)
    if n_negative > 0:
        print(f"sampling {n_negative} no-divergence negative examples...")
        neg_ilocs = sample_negative_ilocs(df, set(event_ilocs), args.window, n_negative, rng)
        for i in neg_ilocs:
            bars = make_bars_window(df, i, args.window)
            if bars is None:
                continue
            examples.append(build_example(bars, None, None, SIGNAL_HOLD, df["timestamp"].iloc[i]))

    examples.sort(key=lambda e: e["_anchor_timestamp"])
    for e in examples:
        del e["_anchor_timestamp"]

    n_val = max(1, int(len(examples) * args.val_fraction)) if examples else 0
    train_examples = examples[: len(examples) - n_val]
    val_examples = examples[len(examples) - n_val :]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with open(train_path, "w") as f:
        for e in train_examples:
            f.write(json.dumps(e) + "\n")
    with open(val_path, "w") as f:
        for e in val_examples:
            f.write(json.dumps(e) + "\n")

    print(f"total examples: {len(examples)} (skipped {skipped_short_history} for insufficient trailing history)")
    print(f"train: {len(train_examples)} -> {train_path}")
    print(f"val:   {len(val_examples)} -> {val_path}")


if __name__ == "__main__":
    main()
