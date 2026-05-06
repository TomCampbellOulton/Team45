# Predicts open, high, low, close RELATIVE TO CURRENT OPEN (multi-output GRU)
# N_STEPS=0  → nowcast  |  N_STEPS≥1 → N-step ahead

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from upgraded_utilities import get_data, feature_engineering, rmse, get_features_and_labels
from model_utils import build_targets, build_result_df, compute_rmse, build_result_df_chained, TARGET_COLS, generate_trading_dates, generate_future_dates


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=4):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class MinMaxScaler:
    def fit(self, data: np.ndarray):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.max - self.min + 1e-8) + self.min


def create_sequences(features, labels, seq_length, n_steps):
    X, y = [], []
    for i in range(len(features) - seq_length - max(n_steps - 1, 0)):
        X.append(features[i: i + seq_length])
        y.append(labels[i + seq_length + n_steps - 1])
    return np.array(X), np.array(y)


def train_model(model, loader, criterion, optimiser, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for feature_batch, label_batch in loader:
            preds = model(feature_batch)
            loss  = criterion(preds, label_batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}    Loss: {epoch_loss:.6f}")


def evaluate(model, feature_test, label_test):
    model.eval()
    with torch.no_grad():
        predictions = model(feature_test).numpy()
    score = rmse(label_test.numpy(), predictions)
    print(f"Test RMSE (scaled): {score:.6f}")
    return predictions, score


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         N_STEPS=1,
         rmse_mode="price",
         FUTURE_STEPS=None,
         return_metrics=False):
    # ── Resolve FUTURE_STEPS ────────────────────────────────────────────
    if FUTURE_STEPS is None:
        FUTURE_STEPS = N_STEPS
    
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df_before_targets = feature_engineering(data)        # ← use df, not data
    df = build_targets(df_before_targets, N_STEPS)

    features, labels = get_features_and_labels(df, TARGET_COLS)   # numpy arrays

    SEQUENCE_LENGTH = 60
    HIDDEN_SIZE = 64
    NUM_LAYERS = 4
    EPOCHS = 30
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    OUTPUT_SIZE = len(TARGET_COLS)

    # ── Fit scalers on training data ONLY (no test-set leakage) ────────────────
    n_train = int(len(features) * DATA_SPLIT_RATIOS[0])

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(features[:n_train])
    features_scaled = feature_scaler.transform(features)

    label_scaler = MinMaxScaler()
    label_scaler.fit(labels[:n_train])
    labels_scaled = label_scaler.transform(labels)
    # ───────────────────────────────────────────────────────────────────────────

    features_seq, labels_seq = create_sequences(features_scaled, labels_scaled,
                                                 SEQUENCE_LENGTH, N_STEPS)
    split = int(len(features_seq) * DATA_SPLIT_RATIOS[0])

    feature_train = torch.tensor(features_seq[:split], dtype=torch.float32)
    label_train   = torch.tensor(labels_seq[:split],   dtype=torch.float32)
    feature_test  = torch.tensor(features_seq[split:], dtype=torch.float32)
    label_test    = torch.tensor(labels_seq[split:],   dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(feature_train, label_train)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model     = GRUModel(feature_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, output_size=OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, loader, criterion, optimiser, EPOCHS)
    predictions_scaled, _ = evaluate(model, feature_test, label_test)

    predicted  = label_scaler.inverse_transform(predictions_scaled)
    actual_rel = label_scaler.inverse_transform(label_test.numpy())

    label_start  = split + SEQUENCE_LENGTH + N_STEPS - 1
    actual_opens = df["open"].values[label_start: label_start + len(predicted)]
    # Generate continuous trading dates for predictions (eliminates date gaps)
    dates        = generate_trading_dates(
        df["Date"].values[label_start],
        len(predicted),
        N_STEPS
    )
    seed_open    = df["open"].values[label_start - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actual_rel, actual_opens, mode=rmse_mode)
    print(f"GRU-MultiOut ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    result_df = build_result_df(predicted, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
    
    # Predict future rows if data extends beyond training window
    n_future = FUTURE_STEPS
    if n_future > 0:
        feature_cols  = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c not in TARGET_COLS]
        features_full = df_before_targets[feature_cols].values
        
        features_full_scaled = feature_scaler.transform(features_full)
        dummy = np.zeros((len(features_full_scaled), 4))
        features_full_seq, _ = create_sequences(features_full_scaled, dummy, SEQUENCE_LENGTH, N_STEPS)
        
        if len(features_full_seq) > len(features_seq):
            future_seqs = torch.tensor(features_full_seq[-n_future:], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                future_preds_scaled = model(future_seqs).numpy()
            future_preds = label_scaler.inverse_transform(future_preds_scaled)

            # Generate future target dates starting from where test ended (no overlap)
            last_test_date = result_df.index[-1]
            future_target_dates = generate_future_dates(last_test_date, len(future_preds), N_STEPS)
            
            # Use the last daisy-chained close from test predictions as continuation point
            continuation_close = result_df["daisy_chained_close"].iloc[-1]
            
            future_df = build_result_df(future_preds,
                                        df_before_targets["open"].values[-n_future:],
                                        idx=future_target_dates,   #df_before_targets.index[-n_future:],
                                        seed_open=seed_open, n_steps=N_STEPS,
                                        continuation_close=continuation_close)
            result_df = pd.concat([result_df, future_df], ignore_index=False)
    
    if return_metrics:
        return (result_df, rmse_scores)
    else:
        return result_df
