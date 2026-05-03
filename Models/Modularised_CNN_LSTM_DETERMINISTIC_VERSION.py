# Predicts open, high, low, close RELATIVE TO CURRENT OPEN (deterministic)
# N_STEPS=0  → nowcast  |  N_STEPS≥1 → N-step ahead

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from upgraded_utilities import feature_engineering, walk_forward_validation, get_data, get_features_and_labels, predict_future_rows
from model_utils import build_targets, build_result_df, compute_rmse, TARGET_COLS, generate_trading_dates, generate_future_dates

# Set the seeds to ensure all 'random' aspects are deterministic hence reproducable for others
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNLSTMModel(nn.Module):

    def __init__(self, input_features, seq_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),             nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def create_sequences(features, labels, seq_len, n_steps):
    X, y = [], []
    for i in range(len(features) - seq_len - max(n_steps - 1, 0)):
        X.append(features[i: i + seq_len])
        y.append(labels[i + seq_len + n_steps - 1])
    return np.array(X), np.array(y)


def make_cnn_lstm_predict(n_steps):
    def cnn_lstm_predict(feature_train_data, label_train_data, feature_test_data, k=None):
        set_seed(42)
        seq_len = 20

        # NOTE: feature_train_data / feature_test_data are already standardised by
        # walk_forward_validation — do NOT apply any additional scaling here.
        X_train, y_train = create_sequences(feature_train_data, label_train_data, seq_len, n_steps)
        X_test,  _       = create_sequences(feature_test_data,
                                             np.zeros((len(feature_test_data), 4)), seq_len, n_steps)

        # Handle case where test window is too small to create sequences
        if len(X_train) == 0 or len(X_test) == 0:
            # Return zero predictions with correct shape
            return np.zeros((len(feature_test_data), 4))

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)

        model         = CNNLSTMModel(X_train.shape[2], seq_len)
        optimiser     = optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        loader        = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)

        model.train()
        for _ in range(50):
            for features_b, labels_b in loader:
                loss = loss_function(model(features_b), labels_b)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).numpy()   # (n_test_seq, 4)

        # Pad the front so output length == len(feature_test_data).
        # Forward-fill from the first valid prediction to avoid zero-spikes.
        pad          = seq_len + max(n_steps - 1, 0)
        output       = np.empty((len(feature_test_data), 4))
        output[pad:] = preds
        if len(preds) > 0:
            output[:pad] = preds[0]
        return output

    return cnn_lstm_predict


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         N_STEPS=5,
         rmse_mode="price",
         # ── Evaluation flag ─────────────────────────────────────────────
         FUTURE_STEPS=None,
         return_metrics=False):
    # ── Resolve FUTURE_STEPS ────────────────────────────────────────────
    if FUTURE_STEPS is None:
        FUTURE_STEPS = N_STEPS
    
    set_seed(42)

    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    data_before_targets = feature_engineering(data)
    data = build_targets(data_before_targets, N_STEPS)

    # get_features_and_labels excludes non-numeric and TARGET_COLS columns automatically.
    # walk_forward_validation handles standardisation internally — no pre-scaling needed.
    features, labels = get_features_and_labels(data, TARGET_COLS)

    predicted, actual, _ = walk_forward_validation(
        features, labels, make_cnn_lstm_predict(N_STEPS),
        data_split_ratios=DATA_SPLIT_RATIOS,
    )

    # actual_opens[t] = open price at the reference time for prediction t.
    # Targets were built as value[t+n] / open[t], so open[t] is the correct denominator.
    training_window = int(len(features) * DATA_SPLIT_RATIOS[0])
    actual_opens    = data["open"].values[training_window: training_window + len(predicted)]
    # Generate continuous trading dates for predictions (eliminates date gaps)
    dates           = generate_trading_dates(
        data["Date"].values[training_window],
        len(predicted),
        N_STEPS
    )
    seed_open       = data["open"].values[training_window - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"CNN-LSTM-DET ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    result_df = build_result_df(predicted, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
    
    # Predict future rows if data extends beyond training window
    feature_cols  = [c for c in data_before_targets.select_dtypes(include=[np.number]).columns
                     if c not in TARGET_COLS]
    features_full = data_before_targets[feature_cols].values
    future_preds, future_opens, future_dates = predict_future_rows(
        data_before_targets, data, features_full, features, labels,
        make_cnn_lstm_predict(N_STEPS), N_STEPS,
    )
    if future_preds is not None:
        # Generate future target dates starting from where test ended (no overlap)
        last_test_date = result_df.index[-1]
        future_target_dates = generate_future_dates(last_test_date, len(future_preds), N_STEPS)
        
        # Use the last daisy-chained close from test predictions as continuation point
        # This ensures daisy-chained prices connect seamlessly without jumping
        continuation_close = result_df["daisy_chained_close"].iloc[-1]
        
        future_df = build_result_df(future_preds, future_opens, idx=future_target_dates,
                                    seed_open=seed_open, n_steps=N_STEPS,
                                    continuation_close=continuation_close)
        result_df = pd.concat([result_df, future_df], ignore_index=False)

    if return_metrics:
        return (result_df, rmse_scores)
    else:
        return result_df