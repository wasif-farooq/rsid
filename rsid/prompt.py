"""Single source of truth for how a bar window + divergence event turns into
a chat prompt for Qwen2.5-0.5B-Instruct, and how its completion is parsed.

Used by scripts/build_dataset.py (to build training examples with a known
label) and scripts/infer_stream.py (to build the same prompt shape live and
parse the model's answer) -- if this drifts between the two, the fine-tuned
model sees a different input distribution at inference time than it was
trained on, silently degrading signal quality.
"""

import json

SIGNAL_BUY = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

SYSTEM_PROMPT = (
    "You are a trading signal assistant specialized in hidden RSI divergence "
    "on short-term crypto price action. You will be shown a recent window of "
    "1-second candles with their RSI(14) values, plus a detected divergence "
    "event. Respond with ONLY a compact JSON object: "
    '{"divergence": "<hidden_bullish|hidden_bearish>", "signal": "<BUY|SELL|HOLD>"}. '
    "No other text."
)


def format_window(bars: list[dict]) -> str:
    """bars: list of dicts with keys timestamp, close, rsi (oldest first).

    Compact CSV-style serialization to keep token count low for a small model.
    """
    lines = ["idx,close,rsi"]
    for i, bar in enumerate(bars):
        close = f"{bar['close']:.4f}"
        rsi = f"{bar['rsi']:.2f}" if bar.get("rsi") is not None else "NA"
        lines.append(f"{i},{close},{rsi}")
    return "\n".join(lines)


def format_event(event: dict | None) -> str:
    if event is None:
        return "Detected divergence: none"
    return (
        f"Detected divergence: {event['type']}\n"
        f"RSI at pivot: {event['rsi_value']:.2f} (previous RSI pivot: {event['prev_rsi_value']:.2f})\n"
        f"Price at pivot: {event['price_value']:.4f} (previous price pivot: {event['prev_price_value']:.4f})"
    )


def build_user_message(bars: list[dict], event: dict | None) -> str:
    return (
        f"Recent 1s candles (oldest to newest):\n{format_window(bars)}\n\n"
        f"{format_event(event)}\n\n"
        "What is the trade signal?"
    )


def build_messages(bars: list[dict], event: dict | None) -> list[dict]:
    """Chat-format messages (system + user) for a training/inference example."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(bars, event)},
    ]


def build_completion(divergence_type: str, signal: str) -> str:
    return json.dumps({"divergence": divergence_type, "signal": signal}, separators=(",", ":"))


def parse_completion(text: str) -> dict:
    """Best-effort parse of a model completion back into {divergence, signal}.

    Falls back to regex signal extraction if the model didn't emit valid JSON.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        signal = str(obj.get("signal", "")).upper()
        if signal in (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD):
            return {"divergence": obj.get("divergence"), "signal": signal}
    except (json.JSONDecodeError, AttributeError):
        pass

    upper = text.upper()
    for signal in (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD):
        if signal in upper:
            return {"divergence": None, "signal": signal}
    return {"divergence": None, "signal": SIGNAL_HOLD}
