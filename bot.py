import os
import time
import base64
import logging
import requests
import asyncio
import threading
from datetime import datetime, timedelta

import ta
import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_GROUP_ID = int(os.environ["TELEGRAM_GROUP_ID"])
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
FINNHUB_URL = "https://finnhub.io/api/v1"

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"
RUNPOD_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

# --- Global app reference ---
_app = None
_main_loop = None

# --- Alert storage ---
# Structure: { alert_id: { type, value, last_triggered, last_triggered_value, active } }
active_alerts = {}
alert_id_counter = [0]

# --- User state for multi-step input ---
user_state = {}

COOLDOWN_MINUTES = 5  # Global cooldown


# ─────────────────────────────────────────────
# RUNPOD HELPERS
# ─────────────────────────────────────────────

def submit_runpod_job(pred_len):
    payload = {
        "input": {"ticker": "GC=F", "pred_len": pred_len, "sample_count": 30},
        "policy": {"ttl": 3600000},
    }
    resp = requests.post(RUNPOD_URL, json=payload, headers=RUNPOD_HEADERS)
    resp.raise_for_status()
    return resp.json()["id"]


def poll_runpod_job(job_id, timeout=300, interval=10):
    url = f"{RUNPOD_STATUS_URL}/{job_id}"
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url, headers=RUNPOD_HEADERS)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data["output"]
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Job ended with status: {status}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError("RunPod job timed out")


# ─────────────────────────────────────────────
# SEND HELPERS (thread-safe)
# ─────────────────────────────────────────────

def send_message_sync(chat_id, text):
    future = asyncio.run_coroutine_threadsafe(
        _app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML"),
        _main_loop
    )
    future.result(timeout=30)


def send_photo_sync(chat_id, image_bytes, caption=""):
    future = asyncio.run_coroutine_threadsafe(
        _app.bot.send_photo(chat_id=chat_id, photo=image_bytes, caption=caption),
        _main_loop
    )
    future.result(timeout=30)


# ─────────────────────────────────────────────
# MARKET DATA & INDICATORS
# ─────────────────────────────────────────────

def get_latest_indicators():
    """Fetch XAU/USD M5 candles from Finnhub and calculate indicators."""
    try:
        import pytz
        import time as time_module

        # Finnhub: get last 5 days of M5 candles
        now = int(time_module.time())
        five_days_ago = now - (5 * 24 * 60 * 60)

        resp = requests.get(
            f"{FINNHUB_URL}/forex/candle",
            params={
                "symbol": "OANDA:XAU_USD",
                "resolution": "5",
                "from": five_days_ago,
                "to": now,
                "token": FINNHUB_API_KEY,
            }
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("s") != "ok" or not data.get("c"):
            logger.error(f"Finnhub returned: {data.get('s')}")
            return None

        # Build DataFrame
        df = pd.DataFrame({
            "open":   data["o"],
            "high":   data["h"],
            "low":    data["l"],
            "close":  data["c"],
            "volume": data["v"],
        }, index=pd.to_datetime(data["t"], unit="s", utc=True))

        df = df.sort_index().dropna()

        if len(df) < 30:
            logger.error("Not enough candles from Finnhub")
            return None

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        df["rsi"]  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df["ema9"] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
        df["ema21"]= ta.trend.EMAIndicator(close=close, window=21).ema_indicator()

        macd = ta.trend.MACD(close=close)
        df["macd_hist"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(close=close, window=20)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        df["atr"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        df["vol_avg"]   = volume.rolling(window=20).mean()
        df["vol_ratio"] = volume / df["vol_avg"]

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        def safe(val):
            return float(val) if val is not None and not pd.isna(val) else None

        # Last candle time in ICT (UTC+7)
        try:
            ict = pytz.timezone("Asia/Bangkok")
            last_ts_ict = df.index[-1].astimezone(ict)
            last_candle_time = last_ts_ict.strftime("%b %d, %Y %-I:%M %p ICT")
        except Exception:
            last_candle_time = "Unknown"

        return {
            "price":            safe(latest["close"]),
            "rsi":              safe(latest["rsi"]),
            "ema9":             safe(latest["ema9"]),
            "ema21":            safe(latest["ema21"]),
            "macd_hist":        safe(latest["macd_hist"]),
            "prev_macd_hist":   safe(prev["macd_hist"]),
            "bb_upper":         safe(latest["bb_upper"]),
            "bb_lower":         safe(latest["bb_lower"]),
            "atr":              safe(latest["atr"]),
            "prev_ema9":        safe(prev["ema9"]),
            "prev_ema21":       safe(prev["ema21"]),
            "volume":           safe(latest["volume"]),
            "vol_avg":          safe(latest["vol_avg"]),
            "vol_ratio":        safe(latest["vol_ratio"]),
            "last_candle_time": last_candle_time,
        }
    except Exception as e:
        logger.error(f"Market data error: {e}")
        return None


# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signal(ind, kronos_bias=None):
    if not ind:
        return None

    signals = []
    score = 0

    if ind["rsi"]:
        if ind["rsi"] < 30:
            score += 2
            signals.append(f"😴 RSI: {ind['rsi']:.1f} (OVERSOLD — bullish)")
        elif ind["rsi"] > 70:
            score -= 2
            signals.append(f"🔥 RSI: {ind['rsi']:.1f} (OVERBOUGHT — bearish)")
        elif ind["rsi"] < 50:
            score -= 1
            signals.append(f"📊 RSI: {ind['rsi']:.1f} (below 50)")
        else:
            score += 1
            signals.append(f"📊 RSI: {ind['rsi']:.1f} (above 50)")

    if ind["ema9"] and ind["ema21"] and ind["prev_ema9"] and ind["prev_ema21"]:
        if ind["ema9"] > ind["ema21"] and ind["prev_ema9"] <= ind["prev_ema21"]:
            score += 3
            signals.append("📈 EMA: Bullish crossover just happened!")
        elif ind["ema9"] < ind["ema21"] and ind["prev_ema9"] >= ind["prev_ema21"]:
            score -= 3
            signals.append("📉 EMA: Bearish crossover just happened!")
        elif ind["ema9"] > ind["ema21"]:
            score += 1
            signals.append(f"📈 EMA: Uptrend (9:{ind['ema9']:.1f} &gt; 21:{ind['ema21']:.1f})")
        else:
            score -= 1
            signals.append(f"📉 EMA: Downtrend (9:{ind['ema9']:.1f} &lt; 21:{ind['ema21']:.1f})")

    if ind["macd_hist"] and ind["prev_macd_hist"]:
        if ind["prev_macd_hist"] < 0 and ind["macd_hist"] > 0:
            score += 3
            signals.append("📈 MACD: Bullish crossover!")
        elif ind["prev_macd_hist"] > 0 and ind["macd_hist"] < 0:
            score -= 3
            signals.append("📉 MACD: Bearish crossover!")
        elif ind["macd_hist"] > 0:
            score += 1
            signals.append("📈 MACD: Bullish momentum")
        else:
            score -= 1
            signals.append("📉 MACD: Bearish momentum")

    if ind["bb_upper"] and ind["bb_lower"]:
        if ind["price"] <= ind["bb_lower"]:
            score += 2
            signals.append("📉 Price at lower Bollinger Band (oversold zone)")
        elif ind["price"] >= ind["bb_upper"]:
            score -= 2
            signals.append("📈 Price at upper Bollinger Band (overbought zone)")
        else:
            signals.append("📊 Bollinger: Price within bands")

    if ind["vol_ratio"] and ind["vol_ratio"] >= 2.0:
        signals.append(f"📊 Volume Spike: {ind['vol_ratio']:.1f}x average — big move possible!")

    if kronos_bias:
        if "BULLISH" in kronos_bias.upper():
            score += 2
            signals.append(f"🤖 Kronos AI: {kronos_bias}")
        elif "BEARISH" in kronos_bias.upper():
            score -= 2
            signals.append(f"🤖 Kronos AI: {kronos_bias}")

    if score >= 4:
        direction = "BUY"
        confidence = "HIGH ⚡⚡⚡" if score >= 7 else "MEDIUM ⚡⚡"
    elif score <= -4:
        direction = "SELL"
        confidence = "HIGH ⚡⚡⚡" if score <= -7 else "MEDIUM ⚡⚡"
    else:
        direction = "WAIT"
        confidence = "LOW ⚡"

    atr = ind["atr"] if ind["atr"] else 5.0
    entry = ind["price"]

    if direction == "BUY":
        sl = round(entry - atr, 2)
        tp1 = round(entry + atr, 2)
        tp2 = round(entry + atr * 2, 2)
    elif direction == "SELL":
        sl = round(entry + atr, 2)
        tp1 = round(entry - atr, 2)
        tp2 = round(entry - atr * 2, 2)
    else:
        sl = tp1 = tp2 = None

    return {
        "direction": direction,
        "confidence": confidence,
        "signals": signals,
        "price": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr,
        "last_candle_time": ind.get("last_candle_time", "Unknown"),
    }


def format_signal_message(sig):
    direction_emoji = "✅ BUY 📈" if sig["direction"] == "BUY" else (
        "🔴 SELL 📉" if sig["direction"] == "SELL" else "⏸ WAIT"
    )

    last_candle = sig.get('last_candle_time', 'Unknown')
    lines = [
        "🥇 <b>XAU/USD Trading Signal (M5)</b>",
        "",
        f"💰 Price: <b>${sig['price']:,.2f}</b>",
        f"🕐 Data: <b>{last_candle}</b>",
        "",
        "<b>Indicator Analysis:</b>",
    ]
    for note in sig["signals"]:
        lines.append(f"  {note}")

    lines += [
        "",
        f"<b>Signal: {direction_emoji}</b>",
        f"⚡ Confidence: <b>{sig['confidence']}</b>",
    ]

    if sig["direction"] in ("BUY", "SELL"):
        lines += [
            "",
            f"🎯 Entry: <b>${sig['price']:,.2f}</b>",
            f"🛑 Stop Loss: <b>${sig['sl']:,.2f}</b> ({abs(sig['price'] - sig['sl']):.1f} pts)",
            f"🎯 TP1: <b>${sig['tp1']:,.2f}</b> (+{abs(sig['tp1'] - sig['price']):.1f} pts) — 1:1",
            f"🎯 TP2: <b>${sig['tp2']:,.2f}</b> (+{abs(sig['tp2'] - sig['price']):.1f} pts) — 1:2",
        ]
    else:
        lines.append("\n⚠️ Mixed signals — better to wait for clearer setup.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# FORECAST HELPERS
# ─────────────────────────────────────────────

def format_row_time(time_str):
    try:
        t = datetime.strptime(time_str.replace(" ICT", "").strip(), "%H:%M")
        end = t + timedelta(hours=1)
        return f"{t.strftime('%-I:%M %p')} → {end.strftime('%-I:%M %p')} ICT"
    except Exception:
        return time_str


def format_forecast_message(output, pred_len):
    table = output["table"]
    rows = table["rows"]

    try:
        start_t = datetime.strptime(table["forecast_start"].replace(" ICT", "").strip(), "%H:%M")
        end_t = datetime.strptime(table["forecast_end"].replace(" ICT", "").strip(), "%H:%M")
        end_final = end_t + timedelta(hours=1)
        period = f"{start_t.strftime('%-I:%M %p')} → {end_final.strftime('%-I:%M %p')} ICT"
    except Exception:
        period = f"{table['forecast_start']} → {table['forecast_end']}"

    bias_emoji = "📈" if "BULLISH" in table["bias"].upper() else "📉"

    lines = [
        f"<b>🥇 XAU/USD Forecast — Next {pred_len}h</b>",
        f"💰 Current Price: <b>${table['current_price']:,.2f}</b>",
        f"{bias_emoji} Bias: <b>{table['bias']}</b>",
        f"🕐 Period: <b>{period}</b>",
        "",
        "<b>Hour-by-Hour Breakdown:</b>",
    ]

    for row in rows:
        arrow = "▲" if row["close"] >= row["open"] else "▼"
        lines.append(
            f"{arrow} <b>{format_row_time(row['time'])}</b>\n"
            f"   O:{row['open']:.1f}  H:{row['high']:.1f}  L:{row['low']:.1f}  C:{row['close']:.1f}\n"
            f"   Band: [{row['lower']:.1f} – {row['upper']:.1f}]"
        )

    return "\n".join(lines), table["bias"]


# ─────────────────────────────────────────────
# BACKGROUND TASKS
# ─────────────────────────────────────────────

def run_forecast_thread(chat_id, pred_len):
    try:
        send_message_sync(chat_id, f"⏳ Running {pred_len}h forecast... please wait (2-3 min)")
        job_id = submit_runpod_job(pred_len)
        output = poll_runpod_job(job_id)
        message, _ = format_forecast_message(output, pred_len)
        chart_bytes = base64.b64decode(output["chart_b64"])
        send_message_sync(chat_id, message)
        send_photo_sync(chat_id, chart_bytes, caption=f"📊 Kronos Gold {pred_len}h Forecast")
    except Exception as e:
        send_message_sync(chat_id, f"❌ Error: {str(e)}")


def run_signal_thread(chat_id, use_kronos=True):
    try:
        send_message_sync(chat_id, "⏳ Analyzing market...")

        kronos_bias = None
        if use_kronos:
            send_message_sync(chat_id, "🤖 Getting Kronos AI bias...")
            try:
                job_id = submit_runpod_job(1)
                output = poll_runpod_job(job_id)
                kronos_bias = output["table"]["bias"]
            except Exception:
                kronos_bias = None

        ind = get_latest_indicators()
        if not ind:
            send_message_sync(chat_id, "❌ Could not fetch market data. Market may be closed.")
            return

        sig = generate_signal(ind, kronos_bias)
        message = format_signal_message(sig)
        send_message_sync(chat_id, message)

    except Exception as e:
        send_message_sync(chat_id, f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────
# ALERT MONITORING (every 1 minute)
# ─────────────────────────────────────────────

def check_alerts_thread():
    if not active_alerts:
        return

    try:
        ind = get_latest_indicators()
        if not ind:
            return

        rsi = ind["rsi"]
        vol_ratio = ind["vol_ratio"]
        price = ind["price"]
        now = datetime.utcnow()

        for alert_id, alert in list(active_alerts.items()):
            if not alert["active"]:
                continue

            triggered = False
            current_value = None
            message = ""

            # ── RSI Above ──
            if alert["type"] == "rsi_above" and rsi and rsi >= alert["value"]:
                current_value = round(rsi, 1)
                triggered = True
                message = (
                    f"🚨 <b>RSI ALERT!</b>\n\n"
                    f"🔥 RSI is <b>ABOVE {alert['value']}</b> (Overbought!)\n"
                    f"📊 Current RSI: <b>{rsi:.1f}</b>\n"
                    f"💰 Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"⚠️ Possible reversal — consider SELL\n"
                    f"👉 Tap 📊 Signal for full analysis"
                )

            # ── RSI Below ──
            elif alert["type"] == "rsi_below" and rsi and rsi <= alert["value"]:
                current_value = round(rsi, 1)
                triggered = True
                message = (
                    f"🚨 <b>RSI ALERT!</b>\n\n"
                    f"😴 RSI is <b>BELOW {alert['value']}</b> (Oversold!)\n"
                    f"📊 Current RSI: <b>{rsi:.1f}</b>\n"
                    f"💰 Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"⚡ Possible bounce — consider BUY\n"
                    f"👉 Tap 📊 Signal for full analysis"
                )

            # ── Volume Spike ──
            elif alert["type"] == "volume_spike" and vol_ratio and vol_ratio >= alert["value"]:
                current_value = round(vol_ratio, 2)
                triggered = True
                message = (
                    f"🚨 <b>VOLUME SPIKE ALERT!</b>\n\n"
                    f"📊 Unusual volume detected!\n"
                    f"📈 Volume: <b>{vol_ratio:.1f}x average</b>\n"
                    f"💰 Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"⚡ High activity — possible big move coming!\n"
                    f"👉 Tap 📊 Signal for full analysis"
                )

            if not triggered:
                continue

            # ── Cooldown check (5 minutes) ──
            if alert["last_triggered"]:
                elapsed = (now - alert["last_triggered"]).total_seconds() / 60
                if elapsed < COOLDOWN_MINUTES:
                    continue

            # ── Value change check — only alert if value changed ──
            last_val = alert.get("last_triggered_value")
            if last_val is not None and current_value == last_val:
                continue

            # ── Fire alert ──
            alert["last_triggered"] = now
            alert["last_triggered_value"] = current_value
            send_message_sync(TELEGRAM_GROUP_ID, message)

    except Exception as e:
        logger.error(f"Alert check error: {e}")


# ─────────────────────────────────────────────
# ALERT HELPERS
# ─────────────────────────────────────────────

def add_alert(chat_id, alert_type, value):
    alert_id_counter[0] += 1
    aid = alert_id_counter[0]
    active_alerts[aid] = {
        "type": alert_type,
        "value": value,
        "cooldown_minutes": COOLDOWN_MINUTES,
        "last_triggered": None,
        "last_triggered_value": None,
        "active": True,
        "chat_id": chat_id,
    }
    return aid


def format_alert_label(alert_type, value):
    labels = {
        "rsi_above": f"🔥 RSI Above {value}",
        "rsi_below": f"😴 RSI Below {value}",
        "volume_spike": f"📊 Volume Spike {value}x average",
    }
    return labels.get(alert_type, alert_type)


# ─────────────────────────────────────────────
# MENUS
# ─────────────────────────────────────────────

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Signal (TA only)", callback_data="signal"),
            InlineKeyboardButton("🤖 Signal + AI", callback_data="signal_ai"),
        ],
        [
            InlineKeyboardButton("📈 Forecast 1h", callback_data="forecast_1"),
            InlineKeyboardButton("📈 Forecast 2h", callback_data="forecast_2"),
        ],
        [
            InlineKeyboardButton("📈 Forecast 3h", callback_data="forecast_3"),
            InlineKeyboardButton("📈 Forecast 4h", callback_data="forecast_4"),
        ],
        [
            InlineKeyboardButton("🚨 Set Alert", callback_data="alert_menu"),
            InlineKeyboardButton("📋 My Alerts", callback_data="list_alerts"),
        ],
        [
            InlineKeyboardButton("🔕 Clear All Alerts", callback_data="clear_alerts"),
        ],
    ])


def alert_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔥 RSI Above", callback_data="alert_rsi_above"),
            InlineKeyboardButton("😴 RSI Below", callback_data="alert_rsi_below"),
        ],
        [
            InlineKeyboardButton("📊 Volume Spike (2x)", callback_data="alert_vol_2"),
            InlineKeyboardButton("📊 Volume Spike (3x)", callback_data="alert_vol_3"),
        ],
        [
            InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu"),
        ],
    ])


# ─────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TELEGRAM_GROUP_ID:
        return
    await update.message.reply_text(
        "🥇 <b>Kronos Gold Trading Bot</b>\n\nTap a button to get started:",
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TELEGRAM_GROUP_ID:
        return
    await update.message.reply_text(
        "🥇 <b>Kronos Gold Trading Bot</b>\n\nWhat would you like to do?",
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TELEGRAM_GROUP_ID:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if user_id not in user_state:
        return

    state = user_state[user_id]

    if state["step"] == "waiting_rsi_above":
        try:
            value = float(text)
            if not (0 < value < 100):
                raise ValueError
            add_alert(chat_id, "rsi_above", value)
            del user_state[user_id]
            await update.message.reply_text(
                f"✅ <b>Alert Set!</b>\n\n🔥 Will notify when RSI goes <b>ABOVE {value}</b>\n"
                f"⏱ Cooldown: {COOLDOWN_MINUTES} min\n"
                f"🔄 Only alerts if RSI value changes",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid. Enter a number between 1-99 (e.g. 70)")

    elif state["step"] == "waiting_rsi_below":
        try:
            value = float(text)
            if not (0 < value < 100):
                raise ValueError
            add_alert(chat_id, "rsi_below", value)
            del user_state[user_id]
            await update.message.reply_text(
                f"✅ <b>Alert Set!</b>\n\n😴 Will notify when RSI drops <b>BELOW {value}</b>\n"
                f"⏱ Cooldown: {COOLDOWN_MINUTES} min\n"
                f"🔄 Only alerts if RSI value changes",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid. Enter a number between 1-99 (e.g. 30)")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat.id
    user_id = query.from_user.id
    if chat_id != TELEGRAM_GROUP_ID:
        return

    data = query.data

    if data == "main_menu":
        await query.edit_message_text(
            "🥇 <b>Kronos Gold Trading Bot</b>\n\nWhat would you like to do?",
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )

    elif data == "signal":
        await query.edit_message_text("⏳ Analyzing market indicators...")
        t = threading.Thread(target=run_signal_thread, args=(chat_id, False))
        t.daemon = True
        t.start()

    elif data == "signal_ai":
        await query.edit_message_text("⏳ Analyzing market + calling Kronos AI (2-3 min)...")
        t = threading.Thread(target=run_signal_thread, args=(chat_id, True))
        t.daemon = True
        t.start()

    elif data.startswith("forecast_"):
        pred_len = int(data.split("_")[1])
        await query.edit_message_text(f"⏳ Running {pred_len}h forecast (2-3 min)...")
        t = threading.Thread(target=run_forecast_thread, args=(chat_id, pred_len))
        t.daemon = True
        t.start()

    elif data == "alert_menu":
        await query.edit_message_text(
            "🚨 <b>Set Alert</b>\n\n"
            "Alerts fire every <b>5 minutes</b> if value changes.\n\n"
            "Choose alert type:",
            parse_mode="HTML",
            reply_markup=alert_menu_keyboard(),
        )

    elif data == "alert_rsi_above":
        user_state[user_id] = {"step": "waiting_rsi_above"}
        await query.edit_message_text(
            "🔥 <b>RSI Above Alert</b>\n\n"
            "Type the RSI level (overbought = 70):\n"
            "Example: <code>70</code>",
            parse_mode="HTML",
        )

    elif data == "alert_rsi_below":
        user_state[user_id] = {"step": "waiting_rsi_below"}
        await query.edit_message_text(
            "😴 <b>RSI Below Alert</b>\n\n"
            "Type the RSI level (oversold = 30):\n"
            "Example: <code>30</code>",
            parse_mode="HTML",
        )

    elif data == "alert_vol_2":
        add_alert(chat_id, "volume_spike", 2.0)
        await query.edit_message_text(
            "✅ <b>Volume Spike Alert Set!</b>\n\n"
            "📊 Will notify when volume is <b>2x above average</b>\n"
            f"⏱ Cooldown: {COOLDOWN_MINUTES} min\n"
            "🔄 Only alerts if volume ratio changes",
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )

    elif data == "alert_vol_3":
        add_alert(chat_id, "volume_spike", 3.0)
        await query.edit_message_text(
            "✅ <b>Volume Spike Alert Set!</b>\n\n"
            "📊 Will notify when volume is <b>3x above average</b>\n"
            f"⏱ Cooldown: {COOLDOWN_MINUTES} min\n"
            "🔄 Only alerts if volume ratio changes",
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )

    elif data == "list_alerts":
        if not active_alerts:
            await query.edit_message_text(
                "📋 <b>Active Alerts</b>\n\nNo alerts set.",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        else:
            lines = ["📋 <b>Active Alerts</b>\n"]
            for aid, alert in active_alerts.items():
                if alert["active"]:
                    lines.append(f"#{aid} {format_alert_label(alert['type'], alert['value'])}")
            await query.edit_message_text(
                "\n".join(lines),
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )

    elif data == "clear_alerts":
        active_alerts.clear()
        await query.edit_message_text(
            "🔕 <b>All alerts cleared!</b>",
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    global _app, _main_loop

    _app = Application.builder().token(TELEGRAM_TOKEN).build()

    _app.add_handler(CommandHandler("start", start_command))
    _app.add_handler(CommandHandler("menu", menu_command))
    _app.add_handler(CallbackQueryHandler(button_callback))
    _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    async def post_init(app):
        global _main_loop
        _main_loop = asyncio.get_event_loop()

    _app.post_init = post_init

    # Alert scheduler — every 1 minute
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_alerts_thread, "interval", minutes=1)
    scheduler.start()

    port = int(os.environ.get("PORT", 8080))
    _app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=f"{WEBHOOK_URL}/webhook",
        url_path="/webhook",
    )


if __name__ == "__main__":
    main()