import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Bidirectional, LayerNormalization,
    Multiply, Softmax, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

SEED           = 42
WINDOW         = 16
EPOCHS         = 200
BATCH_SIZE     = 32
USE_VALIDATION = True

START = "2010-01-01"
END   = "2025-12-31"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATA
# ============================================================

def load(ticker):
    df = yf.download(ticker, start=START, end=END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


spx = load("^GSPC")[["Open", "High", "Low", "Close", "Volume"]]
vix = load("^VIX")[["Close"]].rename(columns={"Close": "VIX"})
df  = spx.join(vix, how="inner")


# ============================================================
# FEATURES
# ============================================================

df["ret_1"]   = df["Close"].pct_change()
df["ret_5"]   = df["Close"].pct_change(5)
df["ma_5"]    = df["Close"].rolling(5).mean()
df["ma_20"]   = df["Close"].rolling(20).mean()
df["ma_50"]   = df["Close"].rolling(50).mean()
df["vol_20"]  = df["Close"].rolling(20).std()
df["mom_5"]   = df["Close"] - df["Close"].shift(5)
df["vix_ma5"] = df["VIX"].rolling(5).mean()
df["vix_chg"] = df["VIX"].pct_change()

df.dropna(inplace=True)


# ============================================================
# WEEKLY RESAMPLE
# ============================================================

weekly = df.resample("W").agg({
    "Open":    "first",
    "High":    "max",
    "Low":     "min",
    "Close":   "last",
    "Volume":  "sum",
    "VIX":     "last",
    "ret_1":   "mean",
    "ret_5":   "mean",
    "ma_5":    "last",
    "ma_20":   "last",
    "ma_50":   "last",
    "vol_20":  "mean",
    "mom_5":   "mean",
    "vix_ma5": "last",
    "vix_chg": "mean"
})

weekly.dropna(inplace=True)
weekly = weekly.loc[START:END]

# Raw weekly prices for reconstruction (before converting to returns)
weekly_open_raw  = weekly["Open"].copy()
weekly_high_raw  = weekly["High"].copy()
weekly_low_raw   = weekly["Low"].copy()
weekly_close_raw = weekly["Close"].copy()


# ============================================================
# WEEKLY RETURNS  
# ============================================================

returns = weekly.copy()
for c in ["Open", "High", "Low", "Close"]:
    returns[c] = weekly[c].pct_change()

returns.dropna(inplace=True)


# ============================================================
# SPLIT  
# ============================================================

dates = returns.index
n     = len(dates)

if USE_VALIDATION:
    train_end_idx = int(n * 0.70)
    val_end_idx   = int(n * 0.85)
    train = returns.iloc[:train_end_idx]
    val   = returns.iloc[train_end_idx:val_end_idx]
    test  = returns.iloc[val_end_idx:]

    print(f"Mode  : 70 / 15 / 15  (with validation)")
    print(f"Train : {train.index.min().date()} → {train.index.max().date()}  ({len(train)} weeks)")
    print(f"Val   : {val.index.min().date()}   → {val.index.max().date()}  ({len(val)} weeks)")
    print(f"Test  : {test.index.min().date()}  → {test.index.max().date()}  ({len(test)} weeks)")

else:
    train_end_idx = int(n * 0.85)
    train = returns.iloc[:train_end_idx]
    val   = None
    test  = returns.iloc[train_end_idx:]

    print(f"Mode  : 85 / 15  (no validation)")
    print(f"Train : {train.index.min().date()} → {train.index.max().date()}  ({len(train)} weeks)")
    print(f"Test  : {test.index.min().date()}  → {test.index.max().date()}  ({len(test)} weeks)")


# ============================================================
# SCALE 
# ============================================================

scaler = MinMaxScaler((-1, 1))
scaler.fit(train)

train_sc = scaler.transform(train)
test_sc  = scaler.transform(test)
val_sc   = scaler.transform(val) if USE_VALIDATION else None


# ============================================================
# SEQUENCE BUILDER
# ============================================================

def make_seq(data, dates_index):
    X, y, idx = [], [], []
    for i in range(WINDOW, len(data)):
        X.append(data[i - WINDOW:i])
        y.append(data[i, :4])
        idx.append(dates_index[i])
    return np.array(X), np.array(y), pd.DatetimeIndex(idx)


# Train sequences
X_train, y_train, train_seq_dates = make_seq(train_sc, train.index)

# Val sequences 
if USE_VALIDATION:
    val_data  = np.vstack([train_sc[-WINDOW:], val_sc])
    val_dates = train.index[-WINDOW:].append(val.index)
    X_val, y_val, val_seq_dates = make_seq(val_data, val_dates)
else:
    X_val, y_val = None, None

# Test sequences (prepend tail of val OR train depending on mode)
if USE_VALIDATION:
    test_data       = np.vstack([val_sc[-WINDOW:], test_sc])
    test_dates_full = val.index[-WINDOW:].append(test.index)
else:
    test_data       = np.vstack([train_sc[-WINDOW:], test_sc])
    test_dates_full = train.index[-WINDOW:].append(test.index)

X_test, y_test, test_seq_dates = make_seq(test_data, test_dates_full)

N_FEATURES = X_train.shape[2]

print(f"\nX_train : {X_train.shape}")
if USE_VALIDATION:
    print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")


# ============================================================
# MODEL
# ============================================================

def build_model():
    inp = Input(shape=(WINDOW, N_FEATURES))

    x = Bidirectional(
        LSTM(64, return_sequences=True,
             dropout=0.2, recurrent_dropout=0.1,
             kernel_regularizer=l2(1e-4))
    )(inp)

    x   = LayerNormalization()(x)

    att = Dense(1, activation="tanh")(x)
    att = Softmax(axis=1)(att)
    x   = Multiply()([x, att])
    x   = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    x   = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x   = Dropout(0.3)(x)
    out = Dense(4)(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="huber")
    return model


model = build_model()
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [ReduceLROnPlateau(patience=8, factor=0.5, verbose=1)]

if USE_VALIDATION:
    callbacks.append(
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    )
else:
    callbacks.append(
        EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
    )


# ============================================================
# TRAIN
# ============================================================

fit_kwargs = dict(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

if USE_VALIDATION:
    fit_kwargs["validation_data"] = (X_val, y_val)

history = model.fit(X_train, y_train, **fit_kwargs)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

def inverse_ohlc(x_scaled):
    """Inverse-transform only the first 4 columns (OHLC returns)."""
    pad = np.zeros((len(x_scaled), N_FEATURES))
    pad[:, :4] = x_scaled
    return scaler.inverse_transform(pad)[:, :4]


pred_ret = inverse_ohlc(model.predict(X_test, verbose=0))
true_ret = inverse_ohlc(y_test)


# ============================================================
# PRICE 
# ============================================================

prev_open  = weekly_open_raw.shift(1).reindex(test_seq_dates)
prev_high  = weekly_high_raw.shift(1).reindex(test_seq_dates)
prev_low   = weekly_low_raw.shift(1).reindex(test_seq_dates)
prev_close = weekly_close_raw.shift(1).reindex(test_seq_dates)

pred_open  = prev_open.values  * (1 + pred_ret[:, 0])
pred_high  = prev_high.values  * (1 + pred_ret[:, 1])
pred_low   = prev_low.values   * (1 + pred_ret[:, 2])
pred_close = prev_close.values * (1 + pred_ret[:, 3])

true_open  = prev_open.values  * (1 + true_ret[:, 0])
true_high  = prev_high.values  * (1 + true_ret[:, 1])
true_low   = prev_low.values   * (1 + true_ret[:, 2])
true_close = prev_close.values * (1 + true_ret[:, 3])

mask = (
    ~np.isnan(prev_close.values) &
    ~np.isnan(prev_open.values)  &
    ~np.isnan(prev_high.values)  &
    ~np.isnan(prev_low.values)
)


# ============================================================
# METRICS  
# ============================================================

mae  = mean_absolute_error(true_close[mask], pred_close[mask])
rmse = np.sqrt(mean_squared_error(true_close[mask], pred_close[mask]))
r2   = r2_score(true_close[mask], pred_close[mask])

true_dir = np.sign(true_ret[mask, 3])
pred_dir = np.sign(pred_ret[mask, 3])
dir_acc  = np.mean(true_dir == pred_dir)

print("\n========= TEST SET — CLOSE PRICE METRICS =========")
print(f"MAE          : {mae:,.2f}")
print(f"RMSE         : {rmse:,.2f}")
print(f"R²           : {r2:.4f}")
print(f"Dir Accuracy : {dir_acc:.2%}")

# ============================================================
# PLOT  
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

ohlc_labels = ["Open", "High", "Low", "Close"]
true_prices = [true_open,  true_high,  true_low,  true_close]
pred_prices = [pred_open,  pred_high,  pred_low,  pred_close]

for ax, label, true_p, pred_p in zip(axes, ohlc_labels, true_prices, pred_prices):
    ax.plot(test_seq_dates[mask], true_p[mask],
            label=f"Actual {label}", linewidth=1.5)
    ax.plot(test_seq_dates[mask], pred_p[mask],
            "--", label=f"Predicted {label}", linewidth=1.5)
    ax.set_title(f"S&P 500 Weekly {label} — Test Set")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("LSTM_results.png", dpi=150)
plt.show()


# ============================================================
# NEXT-WEEK PRED
# ============================================================

last_window  = test_sc[-WINDOW:].reshape(1, WINDOW, N_FEATURES)
next_ret_raw = inverse_ohlc(model.predict(last_window, verbose=0))[0]

next_open  = weekly_open_raw.iloc[-1]  * (1 + next_ret_raw[0])
next_high  = weekly_high_raw.iloc[-1]  * (1 + next_ret_raw[1])
next_low   = weekly_low_raw.iloc[-1]   * (1 + next_ret_raw[2])
next_close = weekly_close_raw.iloc[-1] * (1 + next_ret_raw[3])

print(f"\n===== NEXT WEEK FORECAST =====")
print(f"Open  : {next_open:,.2f}")
print(f"High  : {next_high:,.2f}")
print(f"Low   : {next_low:,.2f}")
print(f"Close : {next_close:,.2f}")


# ============================================================
# SAVE CSV
# ============================================================

results = pd.DataFrame({
    "Date":            test_seq_dates[mask],
    "Actual_Open":     true_open[mask].round(2),
    "Predicted_Open":  pred_open[mask].round(2),
    "Actual_High":     true_high[mask].round(2),
    "Predicted_High":  pred_high[mask].round(2),
    "Actual_Low":      true_low[mask].round(2),
    "Predicted_Low":   pred_low[mask].round(2),
    "Actual_Close":    true_close[mask].round(2),
    "Predicted_Close": pred_close[mask].round(2),
})

results.to_csv("LSTM.csv", index=False)

print("\n==============================")
print("FILE SAVED SUCCESSFULLY")
print("==============================")
print("✔ LSTM.csv")
print("✔ LSTM_results.png")
print("==============================")