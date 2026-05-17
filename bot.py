import os
import time
import base64
import threading
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Config from environment variables ---
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

# Allowed chat IDs (personal + group)
ALLOWED_CHAT_IDS = [
    os.environ["TELEGRAM_CHAT_ID"],       # your personal chat
    os.environ["TELEGRAM_GROUP_ID"],      # Gold Prediction group
]

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

COMMANDS = {
    "/f1": 1,
    "/f2": 2,
    "/f3": 3,
    "/f4": 4,
}


def submit_job(pred_len):
    payload = {
        "input": {
            "ticker": "GC=F",
            "pred_len": pred_len,
            "sample_count": 30,
        },
        "policy": {"ttl": 3600000},
    }
    resp = requests.post(RUNPOD_URL, json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["id"]


def poll_job(job_id, timeout=300, interval=10):
    url = f"{RUNPOD_STATUS_URL}/{job_id}"
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data["output"]
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Job {job_id} ended with status: {status}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Job timed out after {timeout}s")


def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    })


def send_photo(chat_id, image_bytes, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    requests.post(url, data={
        "chat_id": chat_id,
        "caption": caption,
    }, files={
        "photo": ("forecast.png", image_bytes, "image/png"),
    })


def format_message(output, pred_len):
    table = output["table"]
    rows = table["rows"]

    lines = [
        f"<b>🥇 XAU/USD Gold Forecast ({pred_len}h)</b>",
        f"Current Price: <b>${table['current_price']:,.2f}</b>",
        f"Bias: <b>{table['bias']}</b>",
        f"Period: {table['forecast_start']} → {table['forecast_end']}",
        "",
        "<b>Hour-by-Hour Forecast:</b>",
    ]

    for row in rows:
        lines.append(
            f"🕐 <b>{row['time']}</b>  "
            f"O:{row['open']:.1f} H:{row['high']:.1f} "
            f"L:{row['low']:.1f} C:{row['close']:.1f}  "
            f"[{row['lower']:.1f} – {row['upper']:.1f}]"
        )

    return "\n".join(lines)


def run_forecast(chat_id, pred_len):
    try:
        send_message(chat_id, f"⏳ Running {pred_len}h forecast... please wait (2-3 min)")
        job_id = submit_job(pred_len)
        output = poll_job(job_id)
        message = format_message(output, pred_len)
        chart_bytes = base64.b64decode(output["chart_b64"])
        send_message(chat_id, message)
        send_photo(chat_id, chart_bytes, caption=f"Kronos Gold {pred_len}h Forecast")
    except Exception as e:
        send_message(chat_id, f"❌ Error: {str(e)}")


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    # Extract message
    message = data.get("message", {})
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = message.get("text", "").strip().lower()

    # Remove bot username suffix if present (e.g. /f1@ddKroGo_bot)
    if "@" in text:
        text = text.split("@")[0]

    # Security: only respond to allowed chats
    if chat_id not in ALLOWED_CHAT_IDS:
        return jsonify({"ok": True})

    # Handle commands
    if text in ("/start", "/help"):
        send_message(chat_id,
            "🥇 <b>Kronos Gold Forecast Bot</b>\n\n"
            "Commands:\n"
            "/f1 — 1 hour forecast\n"
            "/f2 — 2 hour forecast\n"
            "/f3 — 3 hour forecast\n"
            "/f4 — 4 hour forecast\n"
        )
    elif text in COMMANDS:
        pred_len = COMMANDS[text]
        t = threading.Thread(target=run_forecast, args=(chat_id, pred_len))
        t.daemon = True
        t.start()
    else:
        send_message(chat_id, "Unknown command. Send /help for options.")

    return jsonify({"ok": True})


@app.route("/", methods=["GET"])
def health():
    return "Kronos Gold Bot is running!", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)