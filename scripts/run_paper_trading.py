#!/usr/bin/env python3
"""Production paper-trading entrypoint: live HYPEUSDT aggTrade stream -> 1s
bars -> RSI/pivots -> hidden divergence -> fine-tuned Qwen2.5-0.5B trade
signal -> simulated TP/SL position tracking, with crash-safe state,
structured JSONL logs, and automatic websocket reconnect.

Live mode reconstructs 1s bars from raw aggTrade ticks (rsid.bars.
BarAggregator), matching the exact bar granularity the model was fine-tuned
on -- this is deliberate: an earlier version of this script read Binance's
native kline_1m stream directly instead (see rsid.bars.KlineBarTracker,
config.LIVE_KLINE_INTERVAL/LIVE_BAR_SECONDS, still present but currently
unused), which is a real distribution-shift tradeoff for signal quality
since the model has never seen 1-minute bar dynamics. Switch back to that
by swapping BarAggregator for KlineBarTracker in run_live() if needed.

Reuses the same rsid.bars / rsid.indicators / rsid.divergence / rsid.model /
rsid.prompt code paths as scripts/infer_stream.py, plus rsid.execution
(TP/SL engine), rsid.state_store (crash recovery), rsid.feed (reconnect),
and rsid.logging_utils (structured logs) which are new for this script.

Paper trading only -- no real orders are ever placed, no exchange API keys
are used. Position sizing is a fixed USD notional (config.PAPER_TRADE_NOTIONAL_USD)
and only one position is open at a time; new BUY/SELL signals while a
position is open are logged and ignored, never queued or stacked.

Usage:
    python scripts/run_paper_trading.py
    python scripts/run_paper_trading.py --replay data/processed/HYPEUSDT_features_2025-05-30_2025-05-30.parquet --speed max
"""

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rsid.bars import BarAggregator
from rsid.divergence import StreamingDivergenceDetector
from rsid.execution import PaperTradingEngine
from rsid.feed import ReconnectingTradeFeed
from rsid.indicators import IncrementalWilderRSI
from rsid.logging_utils import log_tick, log_trade_event, setup_paper_trading_logging
from rsid.model import generate_signal, load_model
from rsid.state_store import StateStore


class PaperTradingRunner:
    """Owns the per-bar processing pipeline shared by both live and replay
    modes: TP/SL check on the open position, then indicators/divergence/model
    regardless of position state, then signal handling, then heartbeat and
    state-save bookkeeping."""

    def __init__(self, args, tokenizer, model, engine, state_store, tick_logger, trade_logger, console_logger):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.engine = engine
        self.state_store = state_store
        self.tick_logger = tick_logger
        self.trade_logger = trade_logger
        self.console_logger = console_logger

        self.rsi_calc = IncrementalWilderRSI(period=args.rsi_period)
        self.detector = StreamingDivergenceDetector(lookback=args.pivot_lookback)
        self.bar_window = deque(maxlen=args.window)
        self.processing_lock = threading.Lock()
        self.bars_since_save = 0
        self.last_heartbeat_wall = 0.0

    def process_bar(self, second: int, bar: dict, precomputed_rsi: float | None = None) -> None:
        with self.processing_lock:
            ts = datetime.fromtimestamp(second, tz=timezone.utc)

            # 1. Resolve TP/SL/timeout for an already-open position using
            # *this* bar's high/low, before anything else.
            closed_trade = self.engine.on_bar(ts, bar["high"], bar["low"], bar["close"])
            state_dirty = closed_trade is not None
            if closed_trade is not None:
                log_trade_event(self.trade_logger, self.console_logger, "trade_close", **closed_trade)

            # 2. Indicators/divergence/model run regardless of position state
            # -- signals must be generated and logged even when they'll be
            # ignored because a position is already open.
            rsi = precomputed_rsi if precomputed_rsi is not None else self.rsi_calc.update(bar["close"])
            self.bar_window.append({"timestamp": ts, "close": bar["close"], "rsi": rsi})
            event = self.detector.update(ts, bar["high"], bar["low"], bar["close"], rsi)

            if event is not None and len(self.bar_window) == self.bar_window.maxlen:
                log_trade_event(
                    self.trade_logger,
                    self.console_logger,
                    "divergence_event",
                    timestamp=ts,
                    type=event["type"],
                    rsi_value=event["rsi_value"],
                    price_value=event["price_value"],
                )
                try:
                    pred = generate_signal(self.tokenizer, self.model, list(self.bar_window), event)
                except Exception as exc:
                    # Model inference runs inside the websocket callback --
                    # an uncaught exception here (e.g. a CUDA OOM) would
                    # otherwise vanish silently into the feed thread with no
                    # trace, leaving a divergence_event logged with no
                    # signal ever following it and no visible error anywhere.
                    log_trade_event(
                        self.trade_logger,
                        self.console_logger,
                        "inference_error",
                        timestamp=ts,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pred = None

                if pred is not None:
                    result = self.engine.on_signal(pred["signal"], ts, bar["close"], event)
                    log_trade_event(self.trade_logger, self.console_logger, "signal", **result)
                    if result["action"] == "open":
                        state_dirty = True

            # 3. Heartbeat (throttled -- 1 line/sec forever is too noisy).
            self.bars_since_save += 1
            now_wall = time.time()
            if now_wall - self.last_heartbeat_wall >= config.BAR_HEARTBEAT_EVERY_N_SECONDS:
                log_tick(
                    self.tick_logger,
                    self.console_logger,
                    second=second,
                    close=bar["close"],
                    rsi=rsi,
                    position_open=self.engine.position is not None,
                )
                self.last_heartbeat_wall = now_wall

            # 4. Crash-safe state: on every position open/close immediately,
            # else every STATE_SAVE_EVERY_N_BARS bars.
            if state_dirty or self.bars_since_save >= config.STATE_SAVE_EVERY_N_BARS:
                self.state_store.save({"last_processed_bar_second": second, **self.engine.to_state_dict()})
                self.bars_since_save = 0

    def save_final_state(self, second: int) -> None:
        self.state_store.save({"last_processed_bar_second": second, **self.engine.to_state_dict()})


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--rsi-period", type=int, default=config.RSI_PERIOD)
    parser.add_argument("--pivot-lookback", type=int, default=config.PIVOT_LOOKBACK)
    parser.add_argument("--window", type=int, default=config.PROMPT_WINDOW_BARS)
    parser.add_argument("--notional-usd", type=float, default=config.PAPER_TRADE_NOTIONAL_USD)
    parser.add_argument("--take-profit-pct", type=float, default=config.TAKE_PROFIT_PCT)
    parser.add_argument("--stop-loss-pct", type=float, default=config.STOP_LOSS_PCT)
    parser.add_argument("--lookahead-seconds", type=int, default=config.LOOKAHEAD_SECONDS)
    parser.add_argument("--state-path", default=str(config.PAPER_STATE_PATH))
    parser.add_argument("--trade-log-path", default=str(config.PAPER_TRADE_LOG_PATH))
    parser.add_argument("--tick-log-path", default=str(config.PAPER_TICK_LOG_PATH))
    parser.add_argument(
        "--replay",
        default=None,
        help="Path to a features parquet to replay instead of connecting to the live websocket feed",
    )
    parser.add_argument("--speed", choices=["max", "realtime"], default="max", help="Replay speed (ignored for live mode)")
    return parser


def run_replay(args, runner: PaperTradingRunner, state: dict | None) -> None:
    console_logger = runner.console_logger
    df = pd.read_parquet(args.replay)
    resume_after = state.get("last_processed_bar_second") if state else None
    console_logger.info(f"replaying {len(df)} bars from {args.replay} (speed={args.speed}, resume_after={resume_after})")

    prev_second = None
    last_second = None
    for row in df.itertuples(index=False):
        second = int(pd.Timestamp(row.timestamp).timestamp())
        last_second = second
        if resume_after is not None and second <= resume_after:
            continue
        if args.speed == "realtime" and prev_second is not None:
            time.sleep(max(0.0, second - prev_second))
        prev_second = second

        rsi = None if pd.isna(row.rsi) else float(row.rsi)
        bar = {"high": float(row.high), "low": float(row.low), "close": float(row.close)}
        runner.process_bar(second, bar, precomputed_rsi=rsi)

    console_logger.info("replay complete")
    if last_second is not None:
        runner.save_final_state(last_second)


def run_live(args, runner: PaperTradingRunner, state: dict | None, trade_logger, console_logger) -> None:
    if state and state.get("last_processed_bar_second"):
        gap = time.time() - state["last_processed_bar_second"]
        if gap > 5:
            console_logger.warning(
                f"resuming after ~{gap:.0f}s gap -- any TP/SL touches during that downtime were not "
                "observed (acceptable for paper trading; no real capital at risk)"
            )
            log_trade_event(trade_logger, console_logger, "resume_gap", gap_seconds=round(gap, 1))

    shutdown_event = threading.Event()
    aggregator = BarAggregator()

    def on_disconnect(reason, last_frame_age):
        log_trade_event(
            trade_logger, console_logger, "feed_disconnected", reason=reason, last_frame_age_seconds=round(last_frame_age, 1)
        )

    def on_reconnect_attempt(attempt, backoff):
        log_trade_event(trade_logger, console_logger, "feed_reconnect_attempt", attempt=attempt, backoff_seconds=backoff)

    def on_reconnect(outage_seconds):
        log_trade_event(trade_logger, console_logger, "feed_reconnected", outage_seconds=round(outage_seconds, 1))

    def on_trade(data):
        try:
            price, qty, ts_ms = float(data["p"]), float(data["q"]), int(data["T"])
        except (KeyError, TypeError, ValueError):
            return
        for second, bar in aggregator.on_trade(price, qty, ts_ms):
            runner.process_bar(second, bar)

    stream_symbol = args.symbol.lower()
    url = f"{config.FUTURES_WS_BASE}/{stream_symbol}@aggTrade"
    feed = ReconnectingTradeFeed(
        url=url,
        on_trade=on_trade,
        shutdown_event=shutdown_event,
        backoff_initial=config.WS_RECONNECT_BACKOFF_INITIAL_SECONDS,
        backoff_max=config.WS_RECONNECT_BACKOFF_MAX_SECONDS,
        backoff_multiplier=config.WS_RECONNECT_BACKOFF_MULTIPLIER,
        ping_interval=config.WS_PING_INTERVAL_SECONDS,
        ping_timeout=config.WS_PING_TIMEOUT_SECONDS,
        stall_timeout=config.WS_STALL_TIMEOUT_SECONDS,
        on_disconnect=on_disconnect,
        on_reconnect_attempt=on_reconnect_attempt,
        on_reconnect=on_reconnect,
    )

    def handle_signal(signum, frame):
        console_logger.info(f"received signal {signum}, shutting down...")
        shutdown_event.set()
        feed.force_reconnect()  # unblocks a currently-blocked run_forever() promptly

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def ticker():
        while not shutdown_event.is_set():
            shutdown_event.wait(1)
            for second, bar in aggregator.flush_to(int(time.time())):
                runner.process_bar(second, bar)

    threading.Thread(target=ticker, daemon=True, name="bar-ticker").start()

    console_logger.info(f"connecting to {url}")
    feed.run()

    console_logger.info("shutting down: saving final state")
    runner.save_final_state(int(time.time()))


def main() -> None:
    args = build_argparser().parse_args()

    tick_logger, trade_logger, console_logger = setup_paper_trading_logging(
        Path(args.tick_log_path), Path(args.trade_log_path)
    )
    state_store = StateStore(Path(args.state_path))
    state = state_store.load()
    if state:
        console_logger.info(
            f"resumed state: last_processed_bar_second={state.get('last_processed_bar_second')} "
            f"position_open={state.get('position') is not None}"
        )

    engine = PaperTradingEngine.from_state_dict(
        state,
        notional_usd=args.notional_usd,
        tp_pct=args.take_profit_pct,
        sl_pct=args.stop_loss_pct,
        lookahead_bars=args.lookahead_seconds,
    )

    console_logger.info("loading model...")
    tokenizer, model = load_model()
    model.eval()

    runner = PaperTradingRunner(args, tokenizer, model, engine, state_store, tick_logger, trade_logger, console_logger)

    if args.replay:
        run_replay(args, runner, state)
    else:
        run_live(args, runner, state, trade_logger, console_logger)

    logging.shutdown()


if __name__ == "__main__":
    main()
