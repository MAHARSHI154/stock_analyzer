"""
app.py  –  Stock Trend Prediction  (fully corrected version)
=============================================================

Bugs fixed vs. original
------------------------
1.  DataFrame.append() removed in pandas ≥ 2.0
    → replaced with pd.concat()

2.  DATA LEAKAGE – scaler was re-fitted on test data
    → scaler is now fit ONLY on training data; .transform() is used for
      everything else

3.  `scaler = scaler.scale_` overwrote the scaler object with a raw
    numpy array, breaking inverse_transform
    → scale_factor is captured BEFORE the object is reassigned;
      inverse_transform is now used correctly via the scaler object

4.  yfinance ≥ 0.2 returns MultiIndex columns (e.g. ('Close','AAPL'))
    → columns are flattened to single-level immediately after download

5.  Hard-coded end date (2024-10-01)
    → changed to today's date so the app always uses fresh data

6.  Relative static paths were fragile
    → os.path.join(app.static_folder, …) is used throughout

7.  Missing model file crash
    → clear error message if model file is absent (run train_model.py first)

8.  Image browser-caching issue
    → a timestamp query-string is appended to image URLs in the template
"""

import os
import time
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (required for Flask)
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_from_directory, abort

plt.style.use("fivethirtyeight")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_dl_model.keras")
WINDOW     = 100          # look-back window used during training


def get_model():
    """Load (and cache) the Keras model. Raises RuntimeError if absent."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please run  python train_model.py  first to train and save the model."
        )
    return load_model(MODEL_PATH)


# Load once at startup so every request reuses the same object.
try:
    _model = get_model()
except RuntimeError as e:
    _model = None
    print(f"⚠️  WARNING: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance ≥ 0.2."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def save_figure(fig, filename: str) -> str:
    """Save a matplotlib figure to static/ and return the relative path."""
    path = os.path.join(app.static_folder, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return filename          # relative to static/


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return render_template("index.html")

    stock = (request.form.get("stock") or "POWERGRID.NS").strip().upper()

    # ── Download data ────────────────────────────────────────────────────────
    start = dt.datetime(2000, 1, 1)
    end   = dt.datetime.today()          # always fetch up to today

    try:
        df = yf.download(stock, start=start, end=end, progress=False, auto_adjust=True)
        df = flatten_columns(df)         # ← BUG FIX 4: handle MultiIndex columns
    except Exception as exc:
        return render_template("index.html", error=f"Download failed: {exc}")

    if df.empty:
        return render_template(
            "index.html",
            error=f"No data found for '{stock}'. Check the ticker symbol."
        )

    close = df[["Close"]]

    # ── Descriptive statistics ───────────────────────────────────────────────
    data_desc = df.describe()

    # ── Exponential Moving Averages ──────────────────────────────────────────
    ema20  = close["Close"].ewm(span=20,  adjust=False).mean()
    ema50  = close["Close"].ewm(span=50,  adjust=False).mean()
    ema100 = close["Close"].ewm(span=100, adjust=False).mean()
    ema200 = close["Close"].ewm(span=200, adjust=False).mean()

    # ── Train / test split ───────────────────────────────────────────────────
    split        = int(len(close) * 0.70)
    data_train   = close.iloc[:split]
    data_test    = close.iloc[split:]

    # ── Scaling (NO data leakage) ────────────────────────────────────────────
    # BUG FIX 2 & 3: fit scaler ONLY on training data, then transform test data.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)                       # fit on training data ONLY

    # Provide WINDOW days of training context so the first test sequence
    # has a full look-back – this is the correct approach without data leakage.
    past_window_days = data_train.tail(WINDOW)

    # BUG FIX 1: DataFrame.append() removed in pandas 2.0 → use pd.concat()
    final_df   = pd.concat([past_window_days, data_test], ignore_index=True)
    input_data = scaler.transform(final_df)      # BUG FIX 2: transform, NOT fit_transform

    # ── Build test sequences ─────────────────────────────────────────────────
    x_test, y_test = [], []
    for i in range(WINDOW, len(input_data)):
        x_test.append(input_data[i - WINDOW:i, 0])
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test).reshape(-1, WINDOW, 1)
    y_test = np.array(y_test)

    # ── Predict ──────────────────────────────────────────────────────────────
    if _model is None:
        return render_template(
            "index.html",
            error="Model not loaded. Run 'python train_model.py' first."
        )

    y_predicted = _model.predict(x_test, verbose=0)

    # BUG FIX 3: use inverse_transform (scaler object is still intact)
    y_predicted = scaler.inverse_transform(y_predicted).flatten()
    y_test      = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ── Next-day prediction ──────────────────────────────────────────────────
    last_window  = close.tail(WINDOW).values
    last_scaled  = scaler.transform(last_window)
    x_next       = last_scaled.reshape(1, WINDOW, 1)
    next_pred    = scaler.inverse_transform(_model.predict(x_next, verbose=0))
    next_day_price = round(float(next_pred[0][0]), 2)

    # ── Charts ───────────────────────────────────────────────────────────────
    ts = int(time.time())          # cache-buster

    # Plot 1: EMA 20 & 50
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(close["Close"].values, "y",  label="Closing Price", linewidth=1)
    ax1.plot(ema20.values,           "g",  label="EMA 20",        linewidth=1)
    ax1.plot(ema50.values,           "r",  label="EMA 50",        linewidth=1)
    ax1.set_title(f"{stock} – Closing Price with 20 & 50-Day EMA")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price")
    ax1.legend()
    img1 = save_figure(fig1, "ema_20_50.png")

    # Plot 2: EMA 100 & 200
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(close["Close"].values, "y",  label="Closing Price", linewidth=1)
    ax2.plot(ema100.values,          "g",  label="EMA 100",       linewidth=1)
    ax2.plot(ema200.values,          "r",  label="EMA 200",       linewidth=1)
    ax2.set_title(f"{stock} – Closing Price with 100 & 200-Day EMA")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price")
    ax2.legend()
    img2 = save_figure(fig2, "ema_100_200.png")

    # Plot 3: Prediction vs Actual
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(y_test,      "b",  label="Actual Price",    linewidth=1.2)
    ax3.plot(y_predicted, "r",  label="Predicted Price", linewidth=1.2, linestyle="--")
    ax3.set_title(f"{stock} – Actual vs Predicted Price")
    ax3.set_xlabel("Days (test set)")
    ax3.set_ylabel("Price")
    ax3.legend()
    img3 = save_figure(fig3, "stock_prediction.png")

    # ── Save dataset CSV ─────────────────────────────────────────────────────
    safe_ticker  = stock.replace("/", "_")
    csv_filename = f"{safe_ticker}_dataset.csv"
    csv_path     = os.path.join(app.static_folder, csv_filename)
    df.to_csv(csv_path)

    return render_template(
        "index.html",
        stock=stock,
        plot_ema_20_50     = img1,
        plot_ema_100_200   = img2,
        plot_prediction    = img3,
        data_desc          = data_desc.to_html(classes="table table-bordered table-sm"),
        csv_filename       = csv_filename,
        next_day_price     = next_day_price,
        ts                 = ts,           # cache-buster
    )


@app.route("/download/<path:filename>")
def download_file(filename):
    """Serve a file from the static/ folder as a download."""
    # Prevent path traversal attacks
    safe = os.path.basename(filename)
    file_path = os.path.join(app.static_folder, safe)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(app.static_folder, safe, as_attachment=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
