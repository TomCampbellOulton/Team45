# Predicts open, high, low, close RELATIVE TO CURRENT OPEN
# Uses DlyPrcRet as the pattern signal (price-return variant)
# N_STEPS=0  → nowcast  |  N_STEPS≥1 → N-step ahead

import numpy as np
import pandas as pd

from upgraded_utilities import get_data, feature_engineering, walk_forward_validation, normalise_patterns
from model_utils import build_result_df, compute_rmse, build_result_df_chained, generate_trading_dates


def cosine_distance(x1, x2):
    x1, x2 = x1.flatten(), x2.flatten()
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)


def knn_regression(feature_train, label_train, feature_test, k=3):
    """Inverse-distance weighted KNN returning (n_test, 4) predictions."""
    predictions = []
    for test in feature_test:
        distances = [(cosine_distance(feature_train[i], test), label_train[i])
                     for i in range(len(feature_train))]
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        weights   = np.array([1 / (d + 1e-8) for d, _ in k_nearest])
        values    = np.array([label for _, label in k_nearest])   # (k, 4)
        predictions.append(np.sum(weights[:, None] * values, axis=0) / np.sum(weights))
    return np.array(predictions)   # (n_test, 4)


def build_pattern_dataset_ohlc(returns, open_rel, high_rel, low_rel, close_rel,
                                window=20, n_steps=1):
    features, labels = [], []
    for i in range(window, len(returns) - max(n_steps, 1) + 1):
        pattern   = returns[i - window: i]
        label_idx = (i + n_steps - 1) if n_steps > 0 else (i - 1)
        if label_idx < 0 or label_idx >= len(returns):
            continue
        label = np.array([open_rel[label_idx], high_rel[label_idx],
                          low_rel[label_idx],  close_rel[label_idx]])
        # Skip labels with NaN values (from shift operations)
        if not np.isnan(label).any():
            features.append(pattern)
            labels.append(label)
    return np.array(features), np.array(labels)


def build_future_patterns_ohlc(returns, open_rel, high_rel, low_rel, close_rel,
                                window=20, n_steps=1):
    """Build patterns for rows that build_pattern_dataset_ohlc omits (future rows).
    Skip patterns where the label would contain NaN values."""
    features, labels = [], []
    for i in range(len(returns) - n_steps + 1, len(returns)):
        if i >= window:
            pattern   = returns[i - window: i]
            label_idx = i - 1 if n_steps == 0 else min(i, len(open_rel) - 1)
            # Only use labels if they don't contain NaN
            if label_idx < len(open_rel) and not np.isnan(open_rel[label_idx]):
                label = np.array([open_rel[label_idx], high_rel[label_idx],
                                  low_rel[label_idx],  close_rel[label_idx]])
                if not np.isnan(label).any():
                    features.append(pattern)
                    labels.append(label)
    return np.array(features) if features else np.empty((0, window)), \
           np.array(labels) if labels else np.empty((0, 4))


def run_pattern_knn(df, window, k, DATA_SPLIT_RATIOS, n_steps):
    returns   = df["DlyPrcRet"].values
    high_rel  = (df["high"]  / df["open"]).values
    low_rel   = (df["low"]   / df["open"]).values
    close_rel = (df["close"] / df["open"]).values
    open_rel  = (df["open"].shift(-n_steps) / df["open"]).values if n_steps > 0 \
                else np.ones(len(df))
    
    # Forward-fill NaN values in open_rel (last n_steps rows) to maintain pattern coverage
    if n_steps > 0:
        # Make a writable copy before modifying
        open_rel = open_rel.copy()
        # Find last valid open_rel value and forward-fill
        valid_mask = ~np.isnan(open_rel)
        if valid_mask.any():
            last_valid_idx = np.where(valid_mask)[0][-1]
            last_valid_val = open_rel[last_valid_idx]
            open_rel[~valid_mask] = last_valid_val

    features, labels = build_pattern_dataset_ohlc(
        returns, open_rel, high_rel, low_rel, close_rel, window=window, n_steps=n_steps)
    features = normalise_patterns(features)

    predictions, actuals, models_rmse = walk_forward_validation(
        features, labels, knn_regression,
        data_split_ratios=DATA_SPLIT_RATIOS, k=k,
    )
    return models_rmse, predictions, actuals


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         k=2,
         N_STEPS=5,
         rmse_mode="price"):
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df = feature_engineering(data)
    df_full = df.copy()

    pattern_windows  = [5, 10, 20, 50]
    predictions_list = []
    actuals          = None

    for window in pattern_windows:
        _, predictions, actuals = run_pattern_knn(
            df, window=window, k=k,
            DATA_SPLIT_RATIOS=DATA_SPLIT_RATIOS,
            n_steps=N_STEPS,
        )
        predictions_list.append(predictions)

    min_length       = min(len(p) for p in predictions_list)
    predictions_list = [p[-min_length:] for p in predictions_list]
    predicted        = np.mean(predictions_list, axis=0)   # (min_length, 4)
    actuals          = actuals[-min_length:]

    actual_opens = df["open"].values[-len(predicted):]
    # Generate continuous trading dates for predictions (eliminates date gaps)
    # Use the starting date from the last N rows
    start_idx = len(df) - len(predicted)
    dates = generate_trading_dates(
        df["Date"].values[start_idx],
        len(predicted),
        N_STEPS
    )
    seed_open    = df["open"].values[-len(predicted) - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actuals, actual_opens, mode=rmse_mode)
    print(f"Pattern KNN Prices (k={k}, {N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    result_df = build_result_df(predicted, actual_opens, idx=dates,
                                seed_open=seed_open, n_steps=N_STEPS)
    
    # Predict future rows if available
    n_future = len(df_full) - len(df)
    if n_future > 0:
        returns_full   = df_full["DlyPrcRet"].values
        high_rel_full  = (df_full["high"]  / df_full["open"]).values
        low_rel_full   = (df_full["low"]   / df_full["open"]).values
        close_rel_full = (df_full["close"] / df_full["open"]).values
        open_rel_full  = (df_full["open"].shift(-N_STEPS) / df_full["open"]).values if N_STEPS > 0 \
                        else np.ones(len(df_full))
        
        # Forward-fill NaN values in open_rel_full (last N_STEPS rows) to maintain pattern coverage
        if N_STEPS > 0:
            # Make a writable copy before modifying
            open_rel_full = open_rel_full.copy()
            valid_mask = ~np.isnan(open_rel_full)
            if valid_mask.any():
                last_valid_idx = np.where(valid_mask)[0][-1]
                last_valid_val = open_rel_full[last_valid_idx]
                open_rel_full[~valid_mask] = last_valid_val

        future_pred_list = []
        for window in pattern_windows:
            # Train on all available data
            labeled_feat, labeled_lab = build_pattern_dataset_ohlc(
                returns_full, open_rel_full, high_rel_full, low_rel_full, close_rel_full,
                window=window, n_steps=N_STEPS)
            if len(labeled_feat) == 0:
                continue
            labeled_feat = normalise_patterns(labeled_feat)

            # Predict on future rows
            fut_raw, fut_lab = build_future_patterns_ohlc(
                returns_full, open_rel_full, high_rel_full, low_rel_full, close_rel_full,
                window=window, n_steps=N_STEPS)
            if len(fut_raw) == 0:
                continue
            fut_raw = normalise_patterns(fut_raw)

            future_pred_list.append(knn_regression(labeled_feat, labeled_lab, fut_raw, k=k))

        if future_pred_list:
            min_f = min(len(p) for p in future_pred_list)
            future_combined = np.mean([p[-min_f:] for p in future_pred_list], axis=0)
            
            # Use the last daisy-chained close from test predictions as continuation point
            continuation_close = result_df["daisy_chained_close"].iloc[-1]
            
            future_df = build_result_df(future_combined,
                                        df_full["open"].values[-min_f:],
                                        idx=df_full.index[-min_f:],
                                        seed_open=seed_open,
                                        n_steps=N_STEPS,
                                        continuation_close=continuation_close)
            result_df = pd.concat([result_df, future_df], ignore_index=False)
    
    return result_df
