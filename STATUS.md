# rsid project status

_Last updated: 2026-07-04 (mid-day, before a Claude Code session restart)_

## Pipeline

download data -> compute_features -> label_divergence (batch) -> build_dataset -> finetune_qwen -> infer_stream (live)

Model: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA, fine-tuned to emit `{"divergence": "<hidden_bullish|hidden_bearish>", "signal": "<BUY|SELL|HOLD>"}` from a 30-bar 1s-candle window + detected hidden RSI divergence event.

## Data

- `data/dataset/train.jsonl` — 86,307 examples (full)
- `data/dataset/val.jsonl` — 9,589 examples (full)
- `data/dataset/train_subset.jsonl` / `val_subset.jsonl` — 2,000 / 200 example random subsets, used for fast smoke tests
- Full labeled events: `data/processed/HYPEUSDT_2025-05-30_2026-07-03_labeled.parquet` (324,819 divergence events; BUY 13,900 / SELL 13,896 / HOLD 297,023 before hold-cap downsampling)

## What's been done

1. **Smoke-test training run** on the 2k/200 subsets (1 epoch, batch 1, grad-accum 16, no gradient checkpointing, 125 steps). Loss dropped 0.23 -> 0.03, eval token accuracy settled ~97.7%. Confirms the training loop, config, and 4GB-GPU settings all work.
   - Command used:
     ```
     python scripts/finetune_qwen.py --train-path data/dataset/train_subset.jsonl --val-path data/dataset/val_subset.jsonl --epochs 1 --batch-size 1 --grad-accum 16 --no-gradient-checkpointing --logging-steps 5 --eval-steps 20
     ```
   - Checkpoint: `models/qwen2.5-0.5b-hidden-rsid-lora/checkpoint-125`

2. **Merged the LoRA adapter** into base weights: `models/qwen2.5-0.5b-hidden-rsid-merged` (via `python scripts/finetune_qwen.py --merge`).

3. **Wrote `scripts/backtest.py`** — reconstructs the exact train/val split `build_dataset.py` produces (verified: train=86,307 / val=9,589, exact match) but retains per-example `anchor_iloc`/outcome metadata so model BUY/SELL signals can be walked forward through real price using the same TP(2%)/SL(1%)/1hr-lookahead rules as `label_divergence.py`. Also computes a baseline of mechanically trading every real divergence event by ground-truth direction, for comparison.

4. **Backtest results** (1,000 sampled val examples, current subset-trained checkpoint):
   - **Model predicted SELL zero times** across all 1,000 examples — currently a long-only strategy. Confusion: HOLD->HOLD 434, HOLD->BUY (false positive) 207, SELL->HOLD (missed short) 188, BUY->BUY 131, BUY->HOLD (missed long) 40.
   - Model strategy: 338 trades (all BUY), 59.2% win rate, +196% cumulative return (naive per-trade R sum, no compounding).
   - Baseline (trade every real event, long+short): 909 trades, 61.3% win rate, +559% cumulative return.
   - Read: the gap is almost entirely the missing SELL class, not a bad edge on the trades the model does take (59% vs 61% win rate is close). Caveats: no compounding/fees/slippage, assumes independent full-size positions per trade (no overlap accounting) — a signal-quality comparison, not a capital-accurate equity curve.

## Next step (not yet run)

Full training run on the complete dataset, to fix the missing-SELL bias (likely just an artifact of the thin 2k-example smoke-test subset):

```bash
.venv/bin/python scripts/finetune_qwen.py \
  --train-path data/dataset/train.jsonl \
  --val-path data/dataset/val_subset.jsonl \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --no-gradient-checkpointing \
  --logging-steps 50 \
  --eval-steps 500
```

Notes on the change vs. the smoke-test command:
- `--train-path` swapped to the full `train.jsonl` (43x more data, incl. full SELL class).
- `--val-path` intentionally kept on `val_subset.jsonl` (200 rows) rather than the full `val.jsonl` (9,589 rows) — eval time scales with val-set size (~100-265s for 200 rows in the smoke test), so evaluating on the full val set during training would dominate wall-clock time. The real held-out check is `scripts/backtest.py`, run after training.
- `--logging-steps`/`--eval-steps` bumped up (5/20 -> 50/500) since one epoch is now ~5,400 steps instead of 125.

After training completes:
```bash
.venv/bin/python scripts/finetune_qwen.py --merge
.venv/bin/python scripts/backtest.py --sample-size 1000
```
Compare the new confusion matrix / win rate against the numbers above, especially whether SELL predictions appear at all now.

## Live state as of this save (2026-07-04)

- **The full-data training run above is currently in progress**, PID `103520`, started ~4.5h ago (elapsed ~16,785s), 100% GPU util, ~2.7GB/4GB VRAM. It was launched with the *old* version of `finetune_qwen.py` (before the checkpointing change below), so **it will only save a checkpoint once, at the very end of its single epoch** — no intermediate checkpoint exists yet (confirmed: `models/qwen2.5-0.5b-hidden-rsid-lora/` still only has the old smoke-test `checkpoint-8`/`checkpoint-125`). To pause it without losing progress, suspend rather than kill: `kill -STOP 103520` / `kill -CONT 103520`. Killing it outright loses all progress since no checkpoint exists.

- **`scripts/finetune_qwen.py` was updated** (after this run was already launched, so it doesn't affect it) to add:
  - `--save-steps` (default 500) / `--save-total-limit` (default 3): periodic checkpointing instead of only at epoch end.
  - `--resume-from-checkpoint`: pass with no value to auto-resume from the latest checkpoint under the LoRA output dir, or give an explicit path.
  - If the current run (103520) is stopped/lost and restarted fresh, do NOT pass `--resume-from-checkpoint` on the first invocation — the only checkpoints present (`checkpoint-8`/`checkpoint-125`) are from the old 2k-example smoke test on different data, not this run. Only add `--resume-from-checkpoint` on a rerun of the *same* full-data command after it has itself saved a checkpoint.

- **Explored moving training to Google Colab's free tier** (T4, 16GB VRAM) since the model is tiny (0.5B) and the local 4GB GPU run is slow. Found and set up Google's official `colab-mcp` MCP server:
  - Installed `uv`/`uvx` via the official installer (`curl -LsSf https://astral.sh/uv/install.sh | sh`) since system `pip install uv` was blocked by Debian's externally-managed-environment guard.
  - Registered the server: `claude mcp add colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp` — confirmed connected via `claude mcp list`.
  - **Not yet usable in-session** — MCP tool lists load at session start, so a restart is needed before the Colab tools are callable. **This is the reason for the restart happening now.**
  - Next step once restarted: open a Colab notebook in browser (colab.research.google.com, signed in, Runtime -> Change runtime type -> T4 GPU), then have Claude Code use the new colab-mcp tools to upload `data/dataset/*.jsonl` + `rsid/`/`scripts/`/`config.py`, install `requirements.txt`, and run the same full-training command (can likely raise `--batch-size` given 16GB vs 4GB VRAM).
  - Caveat noted: official docs are thin on exact mechanics (no browser extension mentioned, no explicit auth/GPU-quota details) — this is a brand-new (announced ~2026-03) Google project, so expect some rough edges.
