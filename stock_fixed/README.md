# 📈 Stock Trend Prediction — Corrected & Optimised

A Flask + LSTM web app that downloads historical stock data, visualises
Exponential Moving Averages, and predicts future prices.

---

## 🐛 Bugs Fixed

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `app.py` | `DataFrame.append()` removed in pandas ≥ 2.0 | Replaced with `pd.concat()` |
| 2 | `app.py` | **Data leakage** – scaler was re-fitted on test data | Scaler is fit on training data only; `.transform()` is used for everything else |
| 3 | `app.py` | `scaler = scaler.scale_` destroyed the scaler object, breaking `inverse_transform` | `scale_factor` captured before; `scaler.inverse_transform()` used correctly |
| 4 | `app.py` | yfinance ≥ 0.2 returns MultiIndex columns (`('Close','AAPL')`) | Columns flattened with `get_level_values(0)` right after download |
| 5 | `app.py` | End date hard-coded to 2024-10-01 | Changed to `datetime.today()` |
| 6 | `app.py` | Model file `stock_dl_model.h5` not present → instant crash | Clear error message; separate `train_model.py` creates the file |
| 7 | `app.py` | Browser caches old chart images | Timestamp query-string (`?ts=`) appended to image URLs |
| 8 | `app.py` | Path traversal in `/download/<filename>` | `os.path.basename()` sanitisation added |
| 9 | `notebook` | Model was never saved | `model.save('stock_dl_model.keras')` added as last cell |
| 10 | `notebook` | Same data-leakage & MultiIndex bugs as app.py | Same fixes applied |

---

## 📁 Project Structure

```
stock/
├── app.py                  ← Flask web application (fixed)
├── train_model.py          ← One-time model training script (NEW)
├── stock_predictor.ipynb   ← Jupyter notebook (fixed)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── stock_dl_model.keras    ← Generated after running train_model.py
├── static/                 ← Charts & CSV files (auto-generated)
└── templates/
    └── index.html          ← Frontend template (fixed)
```

---

## ⚙️ Setup & Run

### 1. Prerequisites

- Python 3.9 – 3.11  (TensorFlow does not yet support 3.12 on all platforms)
- `pip` (comes with Python)
- Internet connection (to download stock data from Yahoo Finance)

---

### 2. Create a Virtual Environment  *(recommended)*

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3)?**  Install TensorFlow like this instead:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> pip install -r requirements.txt
> ```

---

### 4. Train the Model  *(run once — takes ~5–10 minutes)*

```bash
python train_model.py
```

This will:
- Download AAPL data from Yahoo Finance (2010 → 2024)
- Train a 2-layer LSTM model with early stopping
- Save it as **`stock_dl_model.keras`**
- Print the next-day predicted price

> You only need to do this once.  The saved model is reused by `app.py`.

---

### 5. Start the Flask Web App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

- Type any valid Yahoo Finance ticker (e.g. `AAPL`, `POWERGRID.NS`, `RELIANCE.NS`, `MSFT`)
- Click **Predict**
- View EMA charts, the prediction chart, descriptive stats, and the next-day forecast
- Download the full CSV dataset

---

### 6. (Optional) Jupyter Notebook

```bash
pip install jupyter
jupyter notebook stock_predictor.ipynb
```

Run all cells in order.  The last cell saves `stock_dl_model.keras` which
`app.py` will pick up automatically.

---

## 🔧 Configuration

Edit the top of `train_model.py` to change the training stock or window size:

```python
STOCK  = "AAPL"       # ticker used for training
WINDOW = 60           # look-back window in days
EPOCHS = 50           # maximum training epochs
```

If you change `WINDOW` here, also update `WINDOW = 100` in `app.py` to match.

---

## 🌐 Supported Tickers

Any ticker available on Yahoo Finance works, for example:

| Ticker | Market |
|--------|--------|
| `AAPL` | NASDAQ |
| `MSFT` | NASDAQ |
| `RELIANCE.NS` | NSE India |
| `POWERGRID.NS` | NSE India |
| `TCS.NS` | NSE India |
| `^NSEI` | Nifty 50 index |

---

## ⚠️ Disclaimer

This project is for educational purposes only.  
**Do not use model predictions to make real financial decisions.**
