import runpod
import sys
import os

sys.path.append("/tmp/Kronos")

print("Step 1: Importing libraries...")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytz
import base64
import io

print("Step 2: Importing Kronos...")
from model import Kronos, KronosTokenizer, KronosPredictor

print("Step 3: Loading tokenizer...")
tokenizer = KronosTokenizer.from_pretrained("/app/models/tokenizer")

print("Step 4: Loading model...")
model = Kronos.from_pretrained("/app/models/kronos-base")

print("Step 5: Creating predictor...")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print("Model loaded!")

def get_data(ticker="GC=F", interval="1h", period="60d"):
    hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    raw = yf.Ticker(ticker).history(period=period, interval=interval)
    raw = raw.reset_index()
    df = pd.DataFrame()
    df["timestamps"] = pd.to_datetime(raw["Datetime"]).dt.tz_convert(hanoi_tz)
    df["open"]  = raw["Open"]
    df["high"]  = raw["High"]
    df["low"]   = raw["Low"]
    df["close"] = raw["Close"]
    df = df.dropna().reset_index(drop=True)
    return df

def run_forecast(df, pred_len=4, sample_count=30):
    lookback = 400
    total = len(df)
    x_df = df.loc[total-lookback:total-1, ["open","high","low","close"]].reset_index(drop=True)
    x_ts = df.loc[total-lookback:total-1, "timestamps"].reset_index(drop=True)
    x_ts_naive = x_ts.dt.tz_localize(None)
    last_time = x_ts_naive.iloc[-1]
    future_ts = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=pred_len, freq="1h")
    y_ts = pd.Series(future_ts)
    all_samples = []
    for _ in range(sample_count):
        s = predictor.predict(df=x_df, x_timestamp=x_ts_naive, y_timestamp=y_ts, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1)
        all_samples.append(s["close"].values)
    all_samples = np.array(all_samples)
    mean_close  = all_samples.mean(axis=0)
    upper_close = np.percentile(all_samples, 85, axis=0)
    lower_close = np.percentile(all_samples, 15, axis=0)
    pred_df = predictor.predict(df=x_df, x_timestamp=x_ts_naive, y_timestamp=y_ts, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=sample_count)
    pred_df.index = future_ts
    pred_df["close_mean"]  = mean_close
    pred_df["close_upper"] = upper_close
    pred_df["close_lower"] = lower_close
    return pred_df, future_ts, x_ts_naive

def build_chart(df, pred_df, future_ts):
    recent = df.tail(20).reset_index(drop=True)
    actual_close = list(recent["close"])
    actual_times = recent["timestamps"].dt.tz_localize(None)
    mean_close  = pred_df["close_mean"].values
    upper_close = pred_df["close_upper"].values
    lower_close = pred_df["close_lower"].values
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(20), actual_close, color="blue", linewidth=2, label="Recent Actual (1hr)")
    ax.plot([19, 20], [actual_close[-1], mean_close[0]], color="orange", linewidth=2, linestyle="--")
    forecast_x = range(20, 20 + len(mean_close))
    ax.plot(list(forecast_x), mean_close, color="orange", linewidth=2.5, linestyle="--", label="Kronos Forecast")
    ax.fill_between(list(forecast_x), lower_close, upper_close, alpha=0.25, color="orange", label="Confidence Band")
    ax.plot(list(forecast_x), upper_close, color="orange", linewidth=1, linestyle=":")
    ax.plot(list(forecast_x), lower_close, color="orange", linewidth=1, linestyle=":")
    ax.axvline(x=19, color="gray", linestyle=":", linewidth=1.5)
    ax.text(19.1, min(actual_close), " NOW", fontsize=9, color="gray")
    ax.axvspan(19, 24, alpha=0.06, color="orange")
    ax.annotate(f"${mean_close[-1]:.2f}", xy=(23, mean_close[-1]), xytext=(22.3, mean_close[-1]+1), fontsize=10, color="darkorange", fontweight="bold")
    all_times = list(actual_times) + list(future_ts)
    tick_positions = list(range(0, 24, 2))
    tick_labels = [all_times[i].strftime("%H:%M") if i < len(all_times) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_title("XAU/USD (Gold) - Live 4-Hour Forecast with Confidence Bands (ICT)", fontsize=13)
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def build_table(df, pred_df, future_ts):
    current_price = df["close"].iloc[-1]
    final_mean = pred_df["close_mean"].iloc[-1]
    bias = f"BULLISH +${final_mean - current_price:.2f}" if final_mean > current_price else f"BEARISH -${current_price - final_mean:.2f}"
    rows = []
    for ts, row in zip(future_ts, pred_df.itertuples()):
        rows.append({"time": ts.strftime("%H:%M ICT"), "open": round(row.open, 2), "high": round(row.high, 2), "low": round(row.low, 2), "close": round(row.close_mean, 2), "lower": round(row.close_lower, 2), "upper": round(row.close_upper, 2)})
    return {"current_price": round(current_price, 2), "bias": bias, "forecast_start": future_ts[0].strftime("%H:%M ICT"), "forecast_end": future_ts[-1].strftime("%H:%M ICT"), "rows": rows}

def handler(job):
    try:
        inp = job.get("input", {})
        ticker = inp.get("ticker", "GC=F")
        period = inp.get("period", "60d")
        pred_len = inp.get("pred_len", 4)
        sample_count = inp.get("sample_count", 30)
        print(f"Fetching {ticker}...")
        df = get_data(ticker=ticker, period=period)
        print(f"Got {len(df)} candles. Forecasting...")
        pred_df, future_ts, _ = run_forecast(df, pred_len=pred_len, sample_count=sample_count)
        chart_b64 = build_chart(df, pred_df, future_ts)
        table = build_table(df, pred_df, future_ts)
        print("Done!")
        return {"status": "success", "table": table, "chart_b64": chart_b64}
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
