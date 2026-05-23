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
import joblib
import json
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

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TWELVEDATA_URL = "https://api.twelvedata.com"

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"
RUNPOD_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

# --- Global app reference ---
_app = None
_main_loop = None

# --- Load custom Gold M5 model ---
_model = None
_scaler = None
_feature_cols = None

# --- Load Romeo V8 model ---
_romeo_v8 = None
_romeo_base_features = [
    'SMA_20','SMA_50','EMA_12','EMA_26','RSI','MACD','MACDSignal',
    'BB_Upper','BB_Middle','BB_Lower','ATR','MFI','Volatility',
    'High_Low_Ratio','Close_Open_Ratio','ROC','Momentum','Volume_MA',
    'Volume_Ratio','Price_Change','High_Low_Spread','Body_Size',
    'Upper_Wick','Lower_Wick','Trend_Up','Trend_Down',
    'RSI_Not_Overbought','RSI_Not_Oversold','MACD_Positive',
    'Close_Above_BB_Middle','Quantum_Entropy','Quantum_Phase',
    'Quantum_Amplitude','Wavelet_Energy','Tree_Feature_1','NN_Feature_1',
    'Linear_Feature_1','Distance_Feature_1','Fractal_Dimension',
    'Fractal_Efficiency','Order_Flow','Market_Depth'
]

def load_custom_model():
    global _model, _scaler, _feature_cols, _romeo_v8
    try:
        _model = joblib.load("gold_m5_model.pkl")
        _scaler = joblib.load("gold_m5_scaler.pkl")
        with open("gold_m5_features.json") as f:
            _feature_cols = json.load(f)
        logger.info(f"Custom Gold M5 model loaded: {len(_feature_cols)} features")
    except Exception as e:
        logger.error(f"Could not load custom model: {e}")
        _model = None

    try:
        _romeo_v8 = joblib.load("romeo_v8_model.pkl")
        logger.info(f"Romeo V8 loaded: {len(_romeo_v8['calibrated_models'])} models")
    except Exception as e:
        logger.error(f"Could not load Romeo V8: {e}")
        _romeo_v8 = None

# --- Alert storage ---
# Structure: { alert_id: { type, value, last_triggered, last_triggered_value, active } }
active_alerts = {}
alert_id_counter = [0]

# --- User state for multi-step input ---
user_state = {}

COOLDOWN_MINUTES = 1     # check every 1 minute
MAX_ALERTS_BEFORE_COOLDOWN = 2   # send 2 alerts then 30min cooldown
LONG_COOLDOWN_MINUTES = 30       # 30min cooldown after 2 alerts


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


def poll_runpod_job(job_id, timeout=480, interval=10):  # 8 min timeout
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
    """Fetch XAU/USD M5 candles from TwelveData (real-time, no delay)."""
    try:
        import pytz

        # TwelveData — real-time XAU/USD M5
        resp = requests.get(
            f"{TWELVEDATA_URL}/time_series",
            params={
                "symbol": "XAU/USD",
                "interval": "5min",
                "outputsize": 100,
                "apikey": TWELVEDATA_API_KEY,
                "format": "JSON",
                "timezone": "Asia/Bangkok",
            }
        )
        resp.raise_for_status()
        data = resp.json()

        if "values" not in data:
            logger.error(f"TwelveData error: {data}")
            return None
        rows = data["values"]
        df_temp = pd.DataFrame(rows)
        df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])
        df_temp = df_temp.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close"]:
            if col in df_temp.columns:
                df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")
        if "volume" not in df_temp.columns:
            df_temp["volume"] = 0
        else:
            df_temp["volume"] = pd.to_numeric(df_temp["volume"], errors="coerce").fillna(0)
        df = df_temp.dropna(subset=["open","high","low","close"])
        logger.info(f"TwelveData: {len(df)} candles, latest: {df.index[-1]}")

        rows = data["values"]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={
            "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        if len(df) < 30:
            logger.error("Not enough candles from TwelveData")
            return None

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

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

        if volume.sum() > 0:
            df["vol_avg"]   = volume.rolling(window=20).mean()
            df["vol_ratio"] = volume / df["vol_avg"].replace(0, 1)
        else:
            df["vol_avg"]   = 0
            df["vol_ratio"] = 0

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        def safe(val):
            return float(val) if val is not None and not pd.isna(val) else None

        # TwelveData returns timestamps in account timezone (Asia/Saigon = ICT)
        # No conversion needed — just display directly
        try:
            last_ts = df.index[-1]
            # Already in ICT — just format it
            last_candle_time = last_ts.strftime("%b %d, %Y %-I:%M %p ICT")
            logger.info(f"Last candle ICT: {last_candle_time}")
        except Exception as e:
            logger.error(f"Timestamp error: {e}")
            last_candle_time = str(df.index[-1])

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
            "volume":           safe(latest["volume"]) if "volume" in latest.index else 0,
            "vol_avg":          safe(latest["vol_avg"]) if "vol_avg" in latest.index else 0,
            "vol_ratio":        safe(latest["vol_ratio"]) if "vol_ratio" in latest.index else 0,
            "last_candle_time": last_candle_time,
            "_raw_df": df.copy(),
        }
    except Exception as e:
        logger.error(f"Market data error: {e}")
        return None


# ─────────────────────────────────────────────
# ROMEO V8 FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_romeo_features(df):
    """Build Romeo V8 features from OHLCV dataframe."""
    d = df.copy()
    # Normalize all columns to lowercase first
    d.columns = [c.lower() for c in d.columns]
    # Ensure volume exists
    if "volume" not in d.columns:
        d["volume"] = 0
    # Capitalize OHLCV for Romeo V8
    d = d.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})

    d['SMA_20'] = d['Close'].rolling(20).mean()
    d['SMA_50'] = d['Close'].rolling(50).mean()
    d['EMA_12'] = d['Close'].ewm(span=12, adjust=False).mean()
    d['EMA_26'] = d['Close'].ewm(span=26, adjust=False).mean()

    delta = d['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    d['RSI'] = 100 - (100 / (1 + up.ewm(alpha=1/14, adjust=False).mean() /
                              (down.ewm(alpha=1/14, adjust=False).mean() + 1e-12)))

    d['MACD'] = d['Close'].ewm(span=12, adjust=False).mean() - d['Close'].ewm(span=26, adjust=False).mean()
    d['MACDSignal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['BB_Middle'] = d['Close'].rolling(20).mean()
    std = d['Close'].rolling(20).std()
    d['BB_Upper'] = d['BB_Middle'] + 2 * std
    d['BB_Lower'] = d['BB_Middle'] - 2 * std
    d['ATR'] = (d['High'] - d['Low']).rolling(14).mean()
    d['MFI'] = 50

    d['Volatility'] = d['Close'].pct_change().rolling(20).std()
    d['High_Low_Ratio'] = (d['High'] - d['Low']) / (d['Close'] + 1e-12)
    d['Close_Open_Ratio'] = (d['Close'] - d['Open']) / (d['Open'] + 1e-12)
    d['ROC'] = d['Close'].pct_change(10)
    d['Momentum'] = d['Close'] - d['Close'].shift(10)
    d['Volume_MA'] = d['Volume'].rolling(20).mean()
    d['Volume_Ratio'] = d['Volume'] / (d['Volume_MA'] + 1e-12)
    d['Price_Change'] = d['Close'].pct_change()
    d['High_Low_Spread'] = d['High'] - d['Low']
    d['Body_Size'] = abs(d['Close'] - d['Open'])
    d['Upper_Wick'] = d['High'] - pd.concat([d['Close'], d['Open']], axis=1).max(axis=1)
    d['Lower_Wick'] = pd.concat([d['Close'], d['Open']], axis=1).min(axis=1) - d['Low']
    d['Trend_Up'] = (d['Close'] > d['SMA_20']).astype(int)
    d['Trend_Down'] = (d['Close'] < d['SMA_20']).astype(int)
    d['RSI_Not_Overbought'] = (d['RSI'] < 70).astype(int)
    d['RSI_Not_Oversold'] = (d['RSI'] > 30).astype(int)
    d['MACD_Positive'] = (d['MACD'] > d['MACDSignal']).astype(int)
    d['Close_Above_BB_Middle'] = (d['Close'] > d['BB_Middle']).astype(int)

    pct = d['Close'].pct_change().fillna(0)
    vol_pct = d['Close'].pct_change().rolling(20).std().fillna(0)
    d['Quantum_Entropy'] = -(pct * np.log(np.abs(pct) + 1e-10)).rolling(20).sum().fillna(0)
    d['Quantum_Phase'] = np.angle(pct + 1j * vol_pct)
    d['Quantum_Amplitude'] = np.abs(pct + 1j * vol_pct)
    d['Wavelet_Energy'] = d['Close'].rolling(20).var().fillna(0)
    d['Tree_Feature_1'] = d['RSI'] * d['MACD']
    d['NN_Feature_1'] = np.sin(d['Quantum_Phase'])
    d['Linear_Feature_1'] = d['Momentum'] / (d['ATR'] + 1e-10)
    d['Distance_Feature_1'] = d['Volatility'] ** 2
    d['Fractal_Dimension'] = (d['High'] - d['Low']).rolling(20).std().fillna(0)
    d['Fractal_Efficiency'] = (d['Close'] - d['Close'].shift(20)).abs() / ((d['High'] - d['Low']).rolling(20).sum() + 1e-10)
    d['Order_Flow'] = (d['Close'] - d['Open']) * d['Volume']
    d['Market_Depth'] = d['Volume'] / (d['High_Low_Spread'] + 1e-10)

    return d


def get_romeo_v8_signal(ohlcv_df):
    """Get Romeo V8 prediction from OHLCV dataframe."""
    if _romeo_v8 is None:
        return None
    try:
        feat_df = build_romeo_features(ohlcv_df.tail(200))
        feat_df = feat_df.dropna()
        if len(feat_df) < 50:
            return None

        X_base = feat_df[_romeo_base_features].values
        X_scaled = _romeo_v8['scaler'].transform(X_base)
        X_pca = _romeo_v8['pca'].transform(X_scaled)
        X_full = np.hstack([X_base, X_pca])

        base_preds = []
        for name, model in _romeo_v8['calibrated_models'].items():
            try:
                base_preds.append(model.predict_proba(X_full)[:, 1])
            except:
                base_preds.append(np.full(X_full.shape[0], 0.5))

        meta_X = np.column_stack(base_preds)
        pred = _romeo_v8['meta_learner'].predict(meta_X)[-1]
        proba = _romeo_v8['meta_learner'].predict_proba(meta_X)[-1]
        confidence = float(max(proba))

        return {
            "direction": "BUY" if pred == 1 else "SELL",
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Romeo V8 prediction error: {e}")
        return None


# ─────────────────────────────────────────────
# CUSTOM MODEL PREDICTION
# ─────────────────────────────────────────────

def get_custom_model_signal(ind):
    """Run our trained Gold M5 model to get BUY/SELL prediction."""
    if _model is None or _scaler is None or _feature_cols is None:
        return None
    try:
        # Build feature row from current indicators
        row = {}
        for col in _feature_cols:
            row[col] = 0.0  # default

        # Map available indicators to feature columns
        mapping = {
            "rsi_14": ind.get("rsi"),
            "ema_9": ind.get("ema9"),
            "ema_21": ind.get("ema21"),
            "ema_9_21_diff": (ind.get("ema9") or 0) - (ind.get("ema21") or 0),
            "macd_hist": ind.get("macd_hist"),
            "bb_upper": ind.get("bb_upper"),
            "bb_lower": ind.get("bb_lower"),
            "atr_14": ind.get("atr"),
            "vol_ratio": ind.get("vol_ratio"),
        }
        for feat, val in mapping.items():
            if feat in row and val is not None:
                row[feat] = val

        X = pd.DataFrame([row])[_feature_cols]
        X_scaled = _scaler.transform(X)
        pred = _model.predict(X_scaled)[0]
        proba = _model.predict_proba(X_scaled)[0]
        confidence = max(proba)

        direction = "BUY" if pred == 1 else "SELL"
        return {"direction": direction, "confidence": float(confidence)}
    except Exception as e:
        logger.error(f"Custom model prediction error: {e}")
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

    # Volume spike removed - no volume data for XAU/USD from TwelveData

    # Custom Gold M5 model
    custom = get_custom_model_signal(ind)
    if custom:
        conf_pct = int(custom["confidence"] * 100)
        if custom["direction"] == "BUY":
            score += 3
            signals.append(f"🧠 Gold AI Model: BUY ({conf_pct}% confidence)")
        else:
            score -= 3
            signals.append(f"🧠 Gold AI Model: SELL ({conf_pct}% confidence)")

    # Romeo V8 Super Ensemble
    romeo = get_romeo_v8_signal(ind.get("_raw_df")) if ind.get("_raw_df") is not None else None
    if romeo:
        conf_pct = int(romeo["confidence"] * 100)
        if romeo["direction"] == "BUY":
            score += 3
            signals.append(f"👑 Romeo V8 (10-model): BUY ({conf_pct}% confidence)")
        else:
            score -= 3
            signals.append(f"👑 Romeo V8 (10-model): SELL ({conf_pct}% confidence)")

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

    # Skip API call if ALL alerts are in cooldown
    now = datetime.utcnow()
    all_in_cooldown = all(
        alert.get("cooldown_until") and now < alert["cooldown_until"]
        for alert in active_alerts.values()
        if alert["active"]
    )
    if all_in_cooldown:
        return  # No API call needed — all alerts cooling down

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

            # ── Long cooldown check (after 2 alerts) ──
            if alert.get("cooldown_until"):
                if now < alert["cooldown_until"]:
                    continue
                else:
                    # Cooldown ended — reset counter, keep last value
                    alert["alert_count"] = 0
                    alert["cooldown_until"] = None

            # ── 1-minute cooldown between alerts ──
            if alert["last_triggered"]:
                elapsed = (now - alert["last_triggered"]).total_seconds() / 60
                if elapsed < COOLDOWN_MINUTES:
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

            # ── MACD Bullish Crossover ──
            elif alert["type"] == "macd_bull":
                macd_hist = ind.get("macd_hist")
                prev_macd_hist = ind.get("prev_macd_hist")
                if macd_hist and prev_macd_hist and prev_macd_hist < 0 and macd_hist > 0:
                    current_value = round(macd_hist, 4)
                    triggered = True
                    message = (
                        f"🚨 <b>MACD ALERT!</b>\n\n"
                        f"📈 MACD Bullish Crossover detected!\n"
                        f"📊 MACD Hist: <b>{macd_hist:.4f}</b>\n"
                        f"💰 Price: <b>${price:,.2f}</b>\n"
                        f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                        f"⚡ Momentum turning BULLISH — consider BUY!\n"
                        f"👉 Tap 📊 Signal for full analysis"
                    )

            # ── MACD Bearish Crossover ──
            elif alert["type"] == "macd_bear":
                macd_hist = ind.get("macd_hist")
                prev_macd_hist = ind.get("prev_macd_hist")
                if macd_hist and prev_macd_hist and prev_macd_hist > 0 and macd_hist < 0:
                    current_value = round(macd_hist, 4)
                    triggered = True
                    message = (
                        f"🚨 <b>MACD ALERT!</b>\n\n"
                        f"📉 MACD Bearish Crossover detected!\n"
                        f"📊 MACD Hist: <b>{macd_hist:.4f}</b>\n"
                        f"💰 Price: <b>${price:,.2f}</b>\n"
                        f"🕐 {datetime.now().strftime('%-I:%M %p')} ICT\n\n"
                        f"⚠️ Momentum turning BEARISH — consider SELL!\n"
                        f"👉 Tap 📊 Signal for full analysis"
                    )

            if not triggered:
                continue

            alert_count = alert.get("alert_count", 0)

            # ── Value change check — only for Alert 1 (not Alert 2) ──
            last_val = alert.get("last_triggered_value")
            if alert_count == 0:
                # Alert 1 — only fire if value changed from last cycle
                if last_val is not None and current_value == last_val:
                    continue

            # ── Fire alert ──
            alert_count += 1
            alert["alert_count"] = alert_count
            alert["last_triggered"] = now
            alert["last_triggered_value"] = current_value

            count_msg = f"🔔 Alert {alert_count}/{MAX_ALERTS_BEFORE_COOLDOWN}"
            if alert_count >= MAX_ALERTS_BEFORE_COOLDOWN:
                count_msg += f" — 30min cooldown starting"
                alert["cooldown_until"] = now + timedelta(minutes=LONG_COOLDOWN_MINUTES)

            send_message_sync(TELEGRAM_GROUP_ID, message + f"\n\n{count_msg}")

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
        "last_triggered": None,
        "last_triggered_value": None,
        "alert_count": 0,          # how many alerts sent in current cycle
        "cooldown_until": None,    # when long cooldown ends
        "active": True,
        "chat_id": chat_id,
    }
    return aid


def format_alert_label(alert_type, value):
    labels = {
        "rsi_above":   f"🔥 RSI Above {value}",
        "rsi_below":   f"😴 RSI Below {value}",
        "macd_bull":   f"📈 MACD Bullish Crossover",
        "macd_bear":   f"📉 MACD Bearish Crossover",
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
            InlineKeyboardButton("📈 MACD Bullish Cross", callback_data="alert_macd_bull"),
            InlineKeyboardButton("📉 MACD Bearish Cross", callback_data="alert_macd_bear"),
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

    elif data == "alert_macd_bull":
        add_alert(chat_id, "macd_bull", 0)
        await query.edit_message_text(
            "✅ <b>MACD Bullish Alert Set!</b>\n\n"
            "📈 Will notify when MACD crosses <b>ABOVE signal</b>\n"
            "⚡ Momentum turning bullish\n"
            f"⏱ Cooldown: {COOLDOWN_MINUTES} min",
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(),
        )

    elif data == "alert_macd_bear":
        add_alert(chat_id, "macd_bear", 0)
        await query.edit_message_text(
            "✅ <b>MACD Bearish Alert Set!</b>\n\n"
            "📉 Will notify when MACD crosses <b>BELOW signal</b>\n"
            "⚠️ Momentum turning bearish\n"
            f"⏱ Cooldown: {COOLDOWN_MINUTES} min",
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

    # Load custom model
    load_custom_model()

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