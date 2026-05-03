import numpy as np
import pandas as pd

from upgraded_utilities import get_data, feature_engineering, walk_forward_validation, predict_future_rows
from model_utils import build_targets, build_result_df, compute_rmse, build_result_df_chained, TARGET_COLS, generate_trading_dates, generate_future_dates


def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)


def knn_regression(feature_train, label_train, feature_test, k=3):
    predictions = []
    for test in feature_test:
        distances = [(cosine_distance(feature_train[i], test), label_train[i])
                     for i in range(len(feature_train))]
        distances.sort(key=lambda x: x[0])
        k_values = [label for (_, label) in distances[:k]]
        predictions.append(np.mean(k_values, axis=0))
    return np.array(predictions)


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         k=2,
         N_STEPS=5,
         rmse_mode="price",
         # ── Evaluation flag ─────────────────────────────────────────────
         return_metrics=False):
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df_before_targets = feature_engineering(data)
    df = build_targets(df_before_targets, N_STEPS)
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna().reset_index(drop=True)

    feature_cols = ["return_t-1", "return_t-2", "return_t-3",
                    "ma_5", "ma_10", "volatility_5", "momentum_5", "momentum_10"]

    features = df[feature_cols].values
    labels   = df[TARGET_COLS].values

    predicted, actual, _ = walk_forward_validation(
        features, labels, knn_regression,
        data_split_ratios=DATA_SPLIT_RATIOS, k=k,
    )

    training_window = int(len(labels) * DATA_SPLIT_RATIOS[0])
    actual_opens    = df["open"].values[training_window: training_window + len(predicted)]
    # Generate continuous trading dates for predictions (eliminates date gaps)
    dates           = generate_trading_dates(
        df["Date"].values[training_window],
        len(predicted),
        N_STEPS
    )
    seed_open       = df["open"].values[training_window - 1]

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"KNN (k={k}, {N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    result_df = build_result_df(predicted, actual_opens, idx=dates,
                                seed_open=seed_open, n_steps=N_STEPS)
    
    # Predict future rows if data extends beyond training window
    feature_cols  = ["return_t-1", "return_t-2", "return_t-3",
                     "ma_5", "ma_10", "volatility_5", "momentum_5", "momentum_10"]
    features_full = df_before_targets[feature_cols].values
    
    future_preds, future_opens, future_dates = predict_future_rows(
        df_before_targets, df, features_full, features, labels,
        knn_regression, N_STEPS, k=k,
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
    
    return (result_df, rmse_scores) if return_metrics else result_df
