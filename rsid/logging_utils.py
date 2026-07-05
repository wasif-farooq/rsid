"""Structured (JSON-lines) logging for live paper trading.

Two on-disk streams, deliberately kept separate since their consumers
differ:
  - ticks.jsonl:  throttled per-bar heartbeat, for feed-health debugging.
                  Daily-rotated since this grows unbounded over a
                  long-lived process.
  - trades.jsonl: every divergence event, every model signal (including
                  ones ignored because a position was already open), every
                  trade open/close, and every feed disconnect/reconnect.
                  Low volume, never rotated -- this is the file
                  scripts/summarize_paper_trades.py reads.

Plus a human-readable console stream (events/signals/trades and the same
heartbeat line) so an operator watching a terminal has liveness confidence
without 1-line-per-second spam.
"""

import json
import logging
import logging.handlers
from pathlib import Path


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # "logged_at" (wall-clock) is intentionally distinct from any
        # domain "timestamp" field the event payload itself carries (e.g. a
        # bar/signal timestamp, which in --replay mode is historical, not
        # wall-clock) -- using the same key for both would make one
        # silently clobber the other.
        payload = {
            "logged_at": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def _make_file_logger(name: str, path: Path, formatter: logging.Formatter, rotate_daily: bool) -> logging.Logger:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    if rotate_daily:
        handler = logging.handlers.TimedRotatingFileHandler(path, when="midnight", backupCount=14, utc=True)
    else:
        handler = logging.FileHandler(path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_paper_trading_logging(tick_log_path: Path, trade_log_path: Path):
    """Returns (tick_logger, trade_logger, console_logger).

    Also attaches a console handler to the "rsid" package logger (parent of
    rsid.feed/rsid.state_store's own module-level `logging.getLogger(__name__)`
    calls) so websocket connect/disconnect/error messages are actually
    visible instead of being silently dropped -- Python's logging module
    swallows INFO/WARNING calls on any logger with no configured handler
    anywhere in its ancestry, which is otherwise exactly what happens here."""
    formatter = JsonLinesFormatter()
    tick_logger = _make_file_logger("rsid.paper.ticks", tick_log_path, formatter, rotate_daily=True)
    trade_logger = _make_file_logger("rsid.paper.trades", trade_log_path, formatter, rotate_daily=False)

    package_logger = logging.getLogger("rsid")
    package_logger.setLevel(logging.INFO)
    package_logger.handlers.clear()
    package_handler = logging.StreamHandler()
    package_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    package_logger.addHandler(package_handler)

    console_logger = logging.getLogger("rsid.paper.console")
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = False
    console_logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    console_logger.addHandler(console_handler)

    return tick_logger, trade_logger, console_logger


def log_tick(tick_logger: logging.Logger, console_logger: logging.Logger | None, **fields) -> None:
    tick_logger.info({"event": "heartbeat", **fields})
    if console_logger is not None:
        console_logger.info(_format_console("heartbeat", fields))


def log_trade_event(trade_logger: logging.Logger, console_logger: logging.Logger | None, event_type: str, **fields) -> None:
    trade_logger.info({"event": event_type, **fields})
    if console_logger is not None:
        console_logger.info(_format_console(event_type, fields))


def _format_console(event_type: str, fields: dict) -> str:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    return f"[{event_type}] {parts}"
