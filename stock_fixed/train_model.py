"""
train_model.py
==============
Run this ONCE before starting the Flask app to train and save the LSTM model.

Usage:
    python train_model.py

The trained model will be saved as: stock_dl_model.keras
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Configuration ─────────────────────────────────────────────────────────────
STOCK        = "AAPL"
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2024-01-01"
WINDOW       = 60          # look-back window (days)
EPOCHS       = 50
BATCH_SIZE   = 32
MODEL_PATH   = "stock_dl_model.keras"
# ──────────────────────────────────────────────────────────────────────────────


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and return a clean single-level DataFrame."""
    df = yf.download(ticker, start=start, end=end, progress=True, auto_adjust=True)

    # yfinance ≥ 0.2 returns MultiIndex columns when a list of tickers is
    # passed, and sometimes even for a single ticker.  Flatten to one level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}. "
            "Check the ticker symbol and your internet connection."
        )
    return df


def build_sequences(scaled_data: np.ndarray, window: int):
    """Create (X, y) sliding-window sequences from scaled 1-D array."""
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i - window:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)


def build_model(window: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True,  input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


def main():
    # 1. Download data
    print(f"\n[1/5] Downloading {STOCK} data …")
    df = download_data(STOCK, TRAIN_START, TRAIN_END)
    close = df[["Close"]]
    print(f"      Downloaded {len(close)} rows  ({close.index[0].date()} → {close.index[-1].date()})")

    # 2. Train / test split  (BEFORE scaling – avoids data leakage)
    print("\n[2/5] Splitting data …")
    split = int(len(close) * 0.80)
    train_data = close.iloc[:split]
    test_data  = close.iloc[split:]
    print(f"      Train: {len(train_data)} rows | Test: {len(test_data)} rows")

    # 3. Scale using ONLY training statistics
    print("\n[3/5] Scaling …")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)

    # For the test set, prepend the last WINDOW rows of training data so that
    # the first test sequence has a full look-back window – NO re-fitting.
    test_with_context = np.concatenate(
        [scaled_train[-WINDOW:], scaler.transform(test_data)], axis=0
    )

    # 4. Build sequences
    print("\n[4/5] Building sequences …")
    X_train, y_train = build_sequences(scaled_train, WINDOW)
    X_test,  y_test  = build_sequences(test_with_context, WINDOW)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    # 5. Build & train
    print("\n[5/5] Training model …")
    model = build_model(WINDOW)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"\n✅  Model saved → {MODEL_PATH}")

    # Quick evaluation plot
    y_pred = scaler.inverse_transform(model.predict(X_test))
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12, 5))
    plt.plot(y_true,  label="Actual",    color="blue")
    plt.plot(y_pred,  label="Predicted", color="red",  linestyle="--")
    plt.title(f"{STOCK} – Training evaluation")
    plt.xlabel("Days (test set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join("static", "training_eval.png")
    os.makedirs("static", exist_ok=True)
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"📊  Evaluation plot saved → {plot_path}")

    # ── Next-day prediction ──────────────────────────────────────────────────
    last_window = close.iloc[-WINDOW:].values
    last_scaled = scaler.transform(last_window)
    X_next      = last_scaled.reshape(1, WINDOW, 1)
    next_pred   = scaler.inverse_transform(model.predict(X_next))
    print(f"\n📈  Next-day predicted price for {STOCK}: ${next_pred[0][0]:.2f}")


if __name__ == "__main__":
    main()
