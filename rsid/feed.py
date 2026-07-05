"""Reconnect-with-backoff wrapper around websocket.WebSocketApp.

A dropped/degraded connection is recovered automatically without resetting
any of the caller's in-memory state -- BarAggregator, the RSI/pivot/
divergence detectors, and any open PaperTradingEngine position all live in
the same process and are untouched by a reconnect; only the socket object
itself is torn down and rebuilt. During the outage, the caller's own
wall-clock ticker (see rsid.bars.BarAggregator.flush_to, driven by
scripts/run_paper_trading.py's ticker thread) keeps producing gap-filled
flat bars exactly as it already does for silent seconds with no trades.

Liveness is judged two ways: websocket-client's own ping_interval/
ping_timeout keepalive (primary -- detects a truly dead socket), and a
secondary watchdog thread that tracks the last time *any* frame arrived
(message, ping, or pong -- deliberately not "last trade", since the market
can go genuinely quiet without the connection being dead) and force-closes
the socket if that goes stale, triggering the outer reconnect loop.
"""

import json
import logging
import threading
import time

import websocket

logger = logging.getLogger(__name__)


class ReconnectingTradeFeed:
    def __init__(
        self,
        url: str,
        on_trade,
        shutdown_event: threading.Event,
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
        backoff_multiplier: float = 2.0,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        stall_timeout: float = 120.0,
        min_stable_seconds: float = 60.0,
        on_disconnect=None,
        on_reconnect_attempt=None,
        on_reconnect=None,
    ):
        self.url = url
        self.on_trade = on_trade
        self.shutdown_event = shutdown_event
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.backoff_multiplier = backoff_multiplier
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.stall_timeout = stall_timeout
        self.min_stable_seconds = min_stable_seconds
        self.on_disconnect = on_disconnect or (lambda reason, last_frame_age: None)
        self.on_reconnect_attempt = on_reconnect_attempt or (lambda attempt, backoff: None)
        self.on_reconnect = on_reconnect or (lambda outage_seconds: None)

        self._ws = None
        self._ws_lock = threading.Lock()
        self._last_frame_time = time.monotonic()
        self._attempt = 0

    def _bump_liveness(self):
        self._last_frame_time = time.monotonic()

    def _on_message(self, ws, message):
        self._bump_liveness()
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("feed: dropped unparseable message: %r", message[:200])
            return
        self.on_trade(data)

    def _on_ping(self, ws, message):
        self._bump_liveness()

    def _on_pong(self, ws, message):
        self._bump_liveness()

    def _on_error(self, ws, error):
        logger.warning("feed: websocket error: %s", error)

    def _on_close(self, ws, code, msg):
        logger.info("feed: websocket closed: %s %s", code, msg)

    def _on_open(self, ws):
        self._bump_liveness()
        logger.info("feed: connected to %s", self.url)

    def _watchdog(self):
        while not self.shutdown_event.is_set():
            self.shutdown_event.wait(5)
            age = time.monotonic() - self._last_frame_time
            if age > self.stall_timeout:
                logger.warning("feed: stall watchdog tripped (no frame in %.0fs), forcing reconnect", age)
                with self._ws_lock:
                    ws = self._ws
                if ws is not None:
                    ws.close()
                self._last_frame_time = time.monotonic()

    def force_reconnect(self):
        """Test hook: force-close the current connection to exercise the reconnect path."""
        with self._ws_lock:
            ws = self._ws
        if ws is not None:
            ws.close()

    def run(self) -> None:
        watchdog_thread = threading.Thread(target=self._watchdog, daemon=True, name="feed-watchdog")
        watchdog_thread.start()

        backoff = self.backoff_initial
        disconnected_at = None
        while not self.shutdown_event.is_set():
            self._attempt += 1
            if self._attempt > 1:
                self.on_reconnect_attempt(self._attempt, backoff)
                logger.info("feed: reconnect attempt %d (backoff %.1fs)", self._attempt, backoff)

            ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_ping=self._on_ping,
                on_pong=self._on_pong,
            )
            with self._ws_lock:
                self._ws = ws
            self._last_frame_time = time.monotonic()
            connected_at = time.monotonic()

            if disconnected_at is not None:
                self.on_reconnect(time.monotonic() - disconnected_at)
                disconnected_at = None

            ws.run_forever(ping_interval=self.ping_interval, ping_timeout=self.ping_timeout)

            with self._ws_lock:
                self._ws = None

            if self.shutdown_event.is_set():
                break

            uptime = time.monotonic() - connected_at
            last_frame_age = time.monotonic() - self._last_frame_time
            disconnected_at = time.monotonic()
            self.on_disconnect("run_forever returned", last_frame_age)
            logger.warning("feed: disconnected after %.1fs uptime", uptime)

            if uptime >= self.min_stable_seconds:
                backoff = self.backoff_initial
            self.shutdown_event.wait(backoff)
            backoff = min(backoff * self.backoff_multiplier, self.backoff_max)
