import os
import time
import base64
import requests

# --- Config from environment variables ---
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

PAYLOAD = {
    "input": {
        "ticker": "GC=F",
        "pred_len": 4,
        "sample_count": 30,
    },
    "policy": {"ttl": 3600000},
}


def submit_job():
    resp = requests.post(RUNPOD_URL, json=PAYLOAD, headers=HEADERS)
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
        print(f"  Waiting... status={status} ({elapsed}s)")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    })
    resp.raise_for_status()


def send_telegram_photo(image_bytes, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
    }, files={
        "photo": ("forecast.png", image_bytes, "image/png"),
    })
    resp.raise_for_status()


def format_message(output):
    table = output["table"]
    rows = table["rows"]

    lines = [
        f"<b>🥇 XAU/USD Gold Forecast</b>",
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


def main():
    print("Submitting job to RunPod...")
    job_id = submit_job()
    print(f"Job submitted: {job_id}")

    print("Polling for result...")
    output = poll_job(job_id)
    print("Job completed!")

    # Send text message
    message = format_message(output)
    send_telegram_message(message)
    print("Text message sent.")

    # Send chart image
    chart_bytes = base64.b64decode(output["chart_b64"])
    send_telegram_photo(chart_bytes, caption="Kronos Gold Forecast Chart")
    print("Chart image sent.")

    print("Done!")


if __name__ == "__main__":
    main()
