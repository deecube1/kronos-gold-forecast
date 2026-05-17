import os
import time
import base64
import logging
import requests
import threading
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import pandas_ta as ta
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
WEBHOOK_URL = os.environ["WEBHOOK_URL"]  # e.g. https://kronos-gold-forecast.onrender.com

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"
RUNPOD_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

# --- Alert storage (in-memory) ---
# Structure: { alert_id: { type, condition, value, cooldown_minutes, last_triggered, active } }
active_alerts = {}
alert_id_counter = [0]

# --- User state for multi-step input ---
user_state = {}  # { chat_id: { "step": ..., "data": ... } }


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
# MARKET DATA & TECHNICAL ANALYSIS
# ─────────────────────────────────────────────

def get_market_data():
    """Fetch latest XAU/USD data with technical indicators."""
    df = yf.download("GC=F", period="5d", interval="5m", progress=False)
    if df.empty:
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    # Calculate indicators
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["ema9"] = ta.ema(df["Close"], length=9)
    df["ema21"] = ta.ema(df["Close"], length=21)

    macd = ta.macd(df["Close"])
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"] = macd.iloc[:, 2]

    bb = ta.bbands(df["Close"], length=20)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]

    atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    if atr is not None:
        df["atr"] = atr

    return df


def get_latest_indicators():
    """Get latest indicator values."""
    df = get_market_data()
    if df is None:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    return {
        "price": float(latest["Close"]),
        "rsi": float(latest["rsi"]) if not pd.isna(latest["rsi"]) else None,
        "ema9": float(latest["ema9"]) if not pd.isna(latest["ema9"]) else None,
        "ema21": float(latest["ema21"]) if not pd.isna(latest["ema21"]) else None,
        "macd": float(latest["macd"]) if "macd" in latest and not pd.isna(latest["macd"]) else None,
        "macd_signal": float(latest["macd_signal"]) if "macd_signal" in latest and not pd.isna(latest["macd_signal"]) else None,
        "macd_hist": float(latest["macd_hist"]) if "macd_hist" in latest and not pd.isna(latest["macd_hist"]) else None,
        "prev_macd_hist": float(prev["macd_hist"]) if "macd_hist" in prev and not pd.isna(prev["macd_hist"]) else None,
        "bb_upper": float(latest["bb_upper"]) if "bb_upper" in latest and not pd.isna(latest["bb_upper"]) else None,
        "bb_mid": float(latest["bb_mid"]) if "bb_mid" in latest and not pd.isna(latest["bb_mid"]) else None,
        "bb_lower": float(latest["bb_lower"]) if "bb_lower" in latest and not pd.isna(latest["bb_lower"]) else None,
        "atr": float(latest["atr"]) if "atr" in latest and not pd.isna(latest["atr"]) else None,
        "prev_ema9": float(prev["ema9"]) if not pd.isna(prev["ema9"]) else None,
        "prev_ema21": float(prev["ema21"]) if not pd.isna(prev["ema21"]) else None,
    }


# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signal(ind, kronos_bias=None):
    """Generate trading signal from indicators + optional Kronos bias."""
    if not ind:
        return None

    signals = []
    score = 0  # positive = bullish, negative = bearish

    # RSI
    rsi_note = ""
    if ind["rsi"]:
        if ind["rsi"] < 30:
            score += 2
            rsi_note = f"😴 RSI: {ind['rsi']:.1f} (OVERSOLD — bullish)"
            signals.append(("bullish", rsi_note))
        elif ind["rsi"] > 70:
            score -= 2
            rsi_note = f"🔥 RSI: {ind['rsi']:.1f} (OVERBOUGHT — bearish)"
            signals.append(("bearish", rsi_note))
        elif ind["rsi"] < 50:
            score -= 1
            rsi_note = f"📊 RSI: {ind['rsi']:.1f} (below 50 — slight bearish)"
            signals.append(("neutral", rsi_note))
        else:
            score += 1
            rsi_note = f"📊 RSI: {ind['rsi']:.1f} (above 50 — slight bullish)"
            signals.append(("neutral", rsi_note))

    # EMA crossover
    ema_note = ""
    if ind["ema9"] and ind["ema21"]:
        if ind["ema9"] > ind["ema21"] and ind["prev_ema9"] <= ind["prev_ema21"]:
            score += 3
            ema_note = "📈 EMA: Bullish crossover just happened!"
            signals.append(("bullish", ema_note))
        elif ind["ema9"] < ind["ema21"] and ind["prev_ema9"] >= ind["prev_ema21"]:
            score -= 3
            ema_note = "📉 EMA: Bearish crossover just happened!"
            signals.append(("bearish", ema_note))
        elif ind["ema9"] > ind["ema21"]:
            score += 1
            ema_note = f"📈 EMA: Uptrend (9:{ind['ema9']:.1f} > 21:{ind['ema21']:.1f})"
            signals.append(("bullish", ema_note))
        else:
            score -= 1
            ema_note = f"📉 EMA: Downtrend (9:{ind['ema9']:.1f} < 21:{ind['ema21']:.1f})"
            signals.append(("bearish", ema_note))

    # MACD
    macd_note = ""
    if ind["macd"] and ind["macd_signal"]:
        if ind["macd"] > ind["macd_signal"] and ind["prev_macd_hist"] and ind["macd_hist"]:
            if ind["prev_macd_hist"] < 0 and ind["macd_hist"] > 0:
                score += 3
                macd_note = "📈 MACD: Bullish crossover!"
                signals.append(("bullish", macd_note))
            elif ind["macd_hist"] > 0:
                score += 1
                macd_note = f"📈 MACD: Bullish momentum"
                signals.append(("bullish", macd_note))
        else:
            if ind["prev_macd_hist"] and ind["macd_hist"] and ind["prev_macd_hist"] > 0 and ind["macd_hist"] < 0:
                score -= 3
                macd_note = "📉 MACD: Bearish crossover!"
                signals.append(("bearish", macd_note))
            elif ind["macd_hist"] and ind["macd_hist"] < 0:
                score -= 1
                macd_note = "📉 MACD: Bearish momentum"
                signals.append(("bearish", macd_note))

    # Bollinger Bands
    bb_note = ""
    if ind["bb_upper"] and ind["bb_lower"] and ind["bb_mid"]:
        if ind["price"] <= ind["bb_lower"]:
            score += 2
            bb_note = f"📉 Price at lower Bollinger Band (oversold zone)"
            signals.append(("bullish", bb_note))
        elif ind["price"] >= ind["bb_upper"]:
            score -= 2
            bb_note = f"📈 Price at upper Bollinger Band (overbought zone)"
            signals.append(("bearish", bb_note))
        else:
            bb_note = f"📊 Bollinger: Price within bands"
            signals.append(("neutral", bb_note))

    # Kronos bias
    kronos_note = ""
    if kronos_bias:
        if "BULLISH" in kronos_bias.upper():
            score += 2
            kronos_note = f"🤖 Kronos AI: {kronos_bias}"
            signals.append(("bullish", kronos_note))
        elif "BEARISH" in kronos_bias.upper():
            score -= 2
            kronos_note = f"🤖 Kronos AI: {kronos_bias}"
            signals.append(("bearish", kronos_note))

    # Determine signal
    if score >= 4:
        direction = "BUY"
        confidence = "HIGH ⚡⚡⚡" if score >= 7 else "MEDIUM ⚡⚡"
    elif score <= -4:
        direction = "SELL"
        confidence = "HIGH ⚡⚡⚡" if score <= -7 else "MEDIUM ⚡⚡"
    else:
        direction = "WAIT"
        confidence = "LOW ⚡"

    # Entry / SL / TP
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
        "score": score,
        "signals": signals,
        "price": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr,
        "kronos_bias": kronos_bias,
    }


def format_signal_message(sig):
    """Format signal into Telegram message."""
    direction_emoji = "✅ BUY 📈" if sig["direction"] == "BUY" else ("🔴 SELL 📉" if sig["direction"] == "SELL" else "⏸ WAIT")

    lines = [
        "🥇 <b>XAU/USD Trading Signal</b>",
        "",
        f"💰 Price: <b>${sig['price']:,.2f}</b>",
        "",
        "<b>Indicator Analysis:</b>",
    ]

    for sentiment, note in sig["signals"]:
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
# TELEGRAM SEND HELPERS
# ─────────────────────────────────────────────

async def send_group_message(app, text):
    await app.bot.send_message(
        chat_id=TELEGRAM_GROUP_ID,
        text=text,
        parse_mode="HTML",
    )


async def send_group_photo(app, image_bytes, caption=""):
    await app.bot.send_photo(
        chat_id=TELEGRAM_GROUP_ID,
        photo=image_bytes,
        caption=caption,
    )


# ─────────────────────────────────────────────
# FORECAST HELPERS
# ─────────────────────────────────────────────

def format_ampm(time_str):
    try:
        t = datetime.strptime(time_str.replace(" ICT", "").strip(), "%H:%M")
        return t.strftime("%-I:%M %p") + " ICT"
    except Exception:
        return time_str


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

def run_forecast_thread(app, chat_id, pred_len):
    """Run forecast in background thread."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        try:
            await app.bot.send_message(chat_id=chat_id, text=f"⏳ Running {pred_len}h forecast... please wait (2-3 min)")
            job_id = submit_runpod_job(pred_len)
            output = poll_runpod_job(job_id)
            message, _ = format_forecast_message(output, pred_len)
            chart_bytes = base64.b64decode(output["chart_b64"])
            await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
            await app.bot.send_photo(chat_id=chat_id, photo=chart_bytes, caption=f"📊 Kronos Gold {pred_len}h Forecast")
        except Exception as e:
            await app.bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")

    loop.run_until_complete(_run())
    loop.close()


def run_signal_thread(app, chat_id, use_kronos=True):
    """Run signal analysis in background thread."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        try:
            await app.bot.send_message(chat_id=chat_id, text="⏳ Analyzing market... please wait")

            kronos_bias = None
            if use_kronos:
                await app.bot.send_message(chat_id=chat_id, text="🤖 Getting Kronos AI bias...")
                try:
                    job_id = submit_runpod_job(1)
                    output = poll_runpod_job(job_id)
                    kronos_bias = output["table"]["bias"]
                except Exception:
                    kronos_bias = None

            ind = get_latest_indicators()
            if not ind:
                await app.bot.send_message(chat_id=chat_id, text="❌ Could not fetch market data. Market may be closed.")
                return

            sig = generate_signal(ind, kronos_bias)
            message = format_signal_message(sig)
            await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")

        except Exception as e:
            await app.bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")

    loop.run_until_complete(_run())
    loop.close()


# ─────────────────────────────────────────────
# ALERT MONITORING (runs every 1 minute)
# ─────────────────────────────────────────────

def check_alerts(app):
    """Check all active alerts against current market data."""
    if not active_alerts:
        return

    try:
        ind = get_latest_indicators()
        if not ind:
            return

        price = ind["price"]
        rsi = ind["rsi"]
        now = datetime.utcnow()

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _send_alert(alert_id, alert, message):
            alert["last_triggered"] = now
            await app.bot.send_message(
                chat_id=TELEGRAM_GROUP_ID,
                text=message,
                parse_mode="HTML",
            )

        for alert_id, alert in list(active_alerts.items()):
            if not alert["active"]:
                continue

            # Check cooldown
            if alert["last_triggered"]:
                elapsed = (now - alert["last_triggered"]).total_seconds() / 60
                if elapsed < alert["cooldown_minutes"]:
                    continue

            triggered = False
            message = ""

            if alert["type"] == "price_above" and price >= alert["value"]:
                triggered = True
                message = (
                    f"🚨 <b>PRICE ALERT!</b>\n\n"
                    f"📈 XAU/USD crossed <b>ABOVE ${alert['value']:,.2f}</b>\n"
                    f"💰 Current Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"👉 Run /signal for full analysis"
                )

            elif alert["type"] == "price_below" and price <= alert["value"]:
                triggered = True
                message = (
                    f"🚨 <b>PRICE ALERT!</b>\n\n"
                    f"📉 XAU/USD dropped <b>BELOW ${alert['value']:,.2f}</b>\n"
                    f"💰 Current Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"👉 Run /signal for full analysis"
                )

            elif alert["type"] == "rsi_above" and rsi and rsi >= alert["value"]:
                triggered = True
                message = (
                    f"🚨 <b>RSI ALERT!</b>\n\n"
                    f"🔥 RSI crossed <b>ABOVE {alert['value']}</b> (Overbought!)\n"
                    f"📊 Current RSI: <b>{rsi:.1f}</b>\n"
                    f"💰 Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"⚠️ Possible reversal — consider SELL\n"
                    f"👉 Run /signal for full analysis"
                )

            elif alert["type"] == "rsi_below" and rsi and rsi <= alert["value"]:
                triggered = True
                message = (
                    f"🚨 <b>RSI ALERT!</b>\n\n"
                    f"😴 RSI dropped <b>BELOW {alert['value']}</b> (Oversold!)\n"
                    f"📊 Current RSI: <b>{rsi:.1f}</b>\n"
                    f"💰 Price: <b>${price:,.2f}</b>\n"
                    f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                    f"⚡ Possible bounce — consider BUY\n"
                    f"👉 Run /signal for full analysis"
                )

            if triggered:
                loop.run_until_complete(_send_alert(alert_id, alert, message))

        loop.close()

    except Exception as e:
        logger.error(f"Alert check error: {e}")


# ─────────────────────────────────────────────
# MENUS
# ─────────────────────────────────────────────

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Signal", callback_data="signal"),
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
            InlineKeyboardButton("📈 Price Above $", callback_data="alert_price_above"),
            InlineKeyboardButton("📉 Price Below $", callback_data="alert_price_below"),
        ],
        [
            InlineKeyboardButton("🔥 RSI Above", callback_data="alert_rsi_above"),
            InlineKeyboardButton("😴 RSI Below", callback_data="alert_rsi_below"),
        ],
        [
            InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu"),
        ],
    ])


# ─────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TELEGRAM_GROUP_ID:
        return

    await update.message.reply_text(
        "🥇 <b>Kronos Gold Trading Bot</b>\n\n"
        "Tap a button to get started:",
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TELEGRAM_GROUP_ID:
        return

    await update.message.reply_text(
        "🥇 <b>Kronos Gold Trading Bot</b>\n\n"
        "What would you like to do?",
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text input for multi-step alert setup."""
    if update.effective_chat.id != TELEGRAM_GROUP_ID:
        return

    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    if chat_id not in user_state:
        return

    state = user_state[chat_id]

    if state["step"] == "waiting_price_above":
        try:
            value = float(text.replace("$", "").replace(",", ""))
            add_alert(chat_id, "price_above", value, 15)
            del user_state[chat_id]
            await update.message.reply_text(
                f"✅ <b>Alert Set!</b>\n\n"
                f"📈 Will notify when XAU/USD goes <b>ABOVE ${value:,.2f}</b>\n"
                f"⏱ Cooldown: 15 minutes",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid price. Please enter a number like 4600")

    elif state["step"] == "waiting_price_below":
        try:
            value = float(text.replace("$", "").replace(",", ""))
            add_alert(chat_id, "price_below", value, 15)
            del user_state[chat_id]
            await update.message.reply_text(
                f"✅ <b>Alert Set!</b>\n\n"
                f"📉 Will notify when XAU/USD drops <b>BELOW ${value:,.2f}</b>\n"
                f"⏱ Cooldown: 15 minutes",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid price. Please enter a number like 4500")

    elif state["step"] == "waiting_rsi_above":
        try:
            value = float(text)
            if not (0 < value < 100):
                raise ValueError
            add_alert(chat_id, "rsi_above", value, 30)
            del user_state[chat_id]
            await update.message.reply_text(
                f"✅ <b>Alert Set!</b>\n\n"
                f"🔥 Will notify when RSI goes <b>ABOVE {value}</b>\n"
                f"⏱ Cooldown: 30 minutes",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid RSI value. Enter a number between 1-99 (e.g. 70)")

    elif state["step"] == "waiting_rsi_below":
        try:
            value = float(text)
            if not (0 < value < 100):
                raise ValueError
            add_alert(chat_id, "rsi_below", value, 30)
            del user_state[chat_id]
            await update.message.reply_text(
                f"✅ <b>Alert Set!</b>\n\n"
                f"😴 Will notify when RSI drops <b>BELOW {value}</b>\n"
                f"⏱ Cooldown: 30 minutes",
                parse_mode="HTML",
                reply_markup=main_menu_keyboard(),
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid RSI value. Enter a number between 1-99 (e.g. 30)")


# ─────────────────────────────────────────────
# ALERT HELPERS
# ─────────────────────────────────────────────

def add_alert(chat_id, alert_type, value, cooldown_minutes):
    alert_id_counter[0] += 1
    aid = alert_id_counter[0]
    active_alerts[aid] = {
        "type": alert_type,
        "value": value,
        "cooldown_minutes": cooldown_minutes,
        "last_triggered": None,
        "active": True,
        "chat_id": chat_id,
    }
    return aid


def format_alert_type(alert_type, value):
    labels = {
        "price_above": f"📈 Price Above ${value:,.2f}",
        "price_below": f"📉 Price Below ${value:,.2f}",
        "rsi_above": f"🔥 RSI Above {value}",
        "rsi_below": f"😴 RSI Below {value}",
    }
    return labels.get(alert_type, alert_type)


# ─────────────────────────────────────────────
# CALLBACK HANDLER
# ─────────────────────────────────────────────

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat.id
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
        t = threading.Thread(target=run_signal_thread, args=(context.application, chat_id, False))
        t.daemon = True
        t.start()

    elif data == "signal_ai":
        await query.edit_message_text("⏳ Analyzing market + calling Kronos AI...")
        t = threading.Thread(target=run_signal_thread, args=(context.application, chat_id, True))
        t.daemon = True
        t.start()

    elif data.startswith("forecast_"):
        pred_len = int(data.split("_")[1])
        await query.edit_message_text(f"⏳ Running {pred_len}h forecast...")
        t = threading.Thread(target=run_forecast_thread, args=(context.application, chat_id, pred_len))
        t.daemon = True
        t.start()

    elif data == "alert_menu":
        await query.edit_message_text(
            "🚨 <b>Set Alert</b>\n\nChoose alert type:",
            parse_mode="HTML",
            reply_markup=alert_menu_keyboard(),
        )

    elif data == "alert_price_above":
        user_state[chat_id] = {"step": "waiting_price_above"}
        await query.edit_message_text(
            "📈 <b>Price Above Alert</b>\n\n"
            "Type the price level you want to be alerted at:\n"
            "Example: <code>4600</code>",
            parse_mode="HTML",
        )

    elif data == "alert_price_below":
        user_state[chat_id] = {"step": "waiting_price_below"}
        await query.edit_message_text(
            "📉 <b>Price Below Alert</b>\n\n"
            "Type the price level you want to be alerted at:\n"
            "Example: <code>4500</code>",
            parse_mode="HTML",
        )

    elif data == "alert_rsi_above":
        user_state[chat_id] = {"step": "waiting_rsi_above"}
        await query.edit_message_text(
            "🔥 <b>RSI Above Alert</b>\n\n"
            "Type the RSI level (overbought is usually 70):\n"
            "Example: <code>70</code>",
            parse_mode="HTML",
        )

    elif data == "alert_rsi_below":
        user_state[chat_id] = {"step": "waiting_rsi_below"}
        await query.edit_message_text(
            "😴 <b>RSI Below Alert</b>\n\n"
            "Type the RSI level (oversold is usually 30):\n"
            "Example: <code>30</code>",
            parse_mode="HTML",
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
                    label = format_alert_type(alert["type"], alert["value"])
                    cooldown = alert["cooldown_minutes"]
                    lines.append(f"#{aid} {label} (cooldown: {cooldown}min)")
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
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Background alert scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_alerts, "interval", minutes=1, args=[app])
    scheduler.start()

    # Start webhook
    port = int(os.environ.get("PORT", 8080))
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=f"{WEBHOOK_URL}/webhook",
        url_path="/webhook",
    )


if __name__ == "__main__":
    main()