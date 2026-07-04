"""Shared configuration for the hidden-RSI-divergence pipeline.

All tunable thresholds live here so the whole pipeline (download -> features ->
labeling -> dataset -> fine-tune -> live inference) stays in sync.
"""

from pathlib import Path

# --- Symbol / data source ---
SYMBOL = "HYPEUSDT"
MARKET = "futures/um"  # Binance USD-M futures (this symbol doesn't exist on spot)
AGGTRADES_BASE_URL = "https://data.binance.vision/data"
FUTURES_WS_BASE = "wss://fstream.binance.com/ws"
EARLIEST_AVAILABLE_DATE = "2025-05-30"  # first day HYPEUSDT aggTrades dump exists

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATASET_DIR = ROOT_DIR / "data" / "dataset"
MODELS_DIR = ROOT_DIR / "models"

# --- Indicators ---
RSI_PERIOD = 14
PIVOT_LOOKBACK = 5  # bars on each side required to confirm a swing high/low

# --- Forward-return trade outcome labeling (batch/offline only) ---
LOOKAHEAD_SECONDS = 3600  # 1 hour; a 1-2% move can take longer than 1 minute to develop
TAKE_PROFIT_PCT = 0.02  # 2%
STOP_LOSS_PCT = 0.01  # 1%

# --- Dataset construction ---
PROMPT_WINDOW_BARS = 30  # trailing candles serialized into each training example
NEGATIVE_SAMPLE_RATIO = 0.15  # fraction of dataset made of sampled no-divergence HOLD bars
VAL_SPLIT_FRACTION = 0.1  # time-based, last N% of events reserved for validation
HOLD_CAP_RATIO = 2.0  # cap HOLD-labeled divergence events at this multiple of BUY+SELL count

# --- Fine-tuning ---
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER_DIR = MODELS_DIR / "qwen2.5-0.5b-hidden-rsid-lora"
LORA_MERGED_DIR = MODELS_DIR / "qwen2.5-0.5b-hidden-rsid-merged"
