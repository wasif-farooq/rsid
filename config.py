"""Shared configuration for the hidden-RSI-divergence pipeline.

All tunable thresholds live here so the whole pipeline (download -> features ->
labeling -> dataset -> fine-tune -> live inference) stays in sync.
"""

from pathlib import Path

# --- Symbol / data source ---
SYMBOL = "HYPEUSDT"
MARKET = "futures/um"  # Binance USD-M futures (this symbol doesn't exist on spot)
AGGTRADES_BASE_URL = "https://data.binance.vision/data"
# NOTE: "/market" prefix is required here -- the plain documented
# "wss://fstream.binance.com/ws" path completes its handshake but silently
# never delivers any push data in some network environments (confirmed via
# direct testing); this legacy "/market/ws" path form (documented at
# https://developers.binance.com/legacy-docs/derivatives/usds-margined-futures/websocket-market-streams/Kline-Candlestick-Streams)
# reliably delivers real data for both aggTrade and kline streams.
FUTURES_WS_BASE = "wss://fstream.binance.com/market/ws"
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

# --- Live data source ---
# Binance USD-M Futures has no 1s kline -- the smallest native interval is
# 1m (verified against Binance's own docs), so live mode reads Binance's own
# 1-minute candles directly instead of reconstructing 1s bars from aggTrade
# ticks. NOTE: the fine-tuned model was trained on 1-second bar dynamics
# (see scripts/build_dataset.py / PROMPT_WINDOW_BARS); running it on 1-minute
# bars is a real distribution shift the model was never trained for -- this
# is a deliberate accepted tradeoff (fast to wire up, no retrain), not a
# claim that signal quality is preserved. --replay mode is unaffected and
# still operates on the original 1s-bar historical data.
LIVE_KLINE_INTERVAL = "1m"
LIVE_BAR_SECONDS = 60

# --- Live paper trading ---
PAPER_TRADE_NOTIONAL_USD = 100.0  # fixed notional per simulated trade, no account-equity tracking
PAPER_TRADING_DIR = ROOT_DIR / "data" / "paper_trading"
PAPER_STATE_PATH = PAPER_TRADING_DIR / "state.json"  # open-position/crash-recovery snapshot
PAPER_TRADE_LOG_PATH = PAPER_TRADING_DIR / "trades.jsonl"  # events/signals/trade open-close, never rotated
PAPER_TICK_LOG_PATH = PAPER_TRADING_DIR / "ticks.jsonl"  # throttled per-bar heartbeat, daily-rotated

STATE_SAVE_EVERY_N_BARS = 60  # periodic snapshot cadence; also saved immediately on any position open/close
BAR_HEARTBEAT_EVERY_N_SECONDS = 60  # console/tick-log liveness cadence (1 bar/sec forever is too noisy to print raw)

# --- WebSocket robustness (live paper trading only) ---
WS_RECONNECT_BACKOFF_INITIAL_SECONDS = 1.0
WS_RECONNECT_BACKOFF_MAX_SECONDS = 60.0
WS_RECONNECT_BACKOFF_MULTIPLIER = 2.0
WS_PING_INTERVAL_SECONDS = 20  # websocket-client's own keepalive
WS_PING_TIMEOUT_SECONDS = 10
WS_STALL_TIMEOUT_SECONDS = 120  # watchdog: force reconnect if no frame (msg/ping/pong) at all in this long
