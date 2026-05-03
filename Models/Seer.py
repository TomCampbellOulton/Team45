# Seer — ground-truth OHLC for test window + available future data.
# Returns the same DataFrame format as all models for direct comparison.
# Reads actual observed prices from the data.

import numpy as np
import pandas as pd

from upgraded_utilities import get_data, feature_engineering, data_split, get_features_and_labels
from model_utils import build_targets, build_result_df, TARGET_COLS, generate_trading_dates, generate_future_dates


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.0, 0.2),
         N_STEPS=1,
         rmse_mode="price",
         FUTURE_STEPS=None,
         return_metrics=False):
    """
    Seer returns actual observed OHLC prices, formatted identically to model outputs.
    
    For the test window: Returns actual prices relative to each day's opening price
    For future rows: Returns actual observed prices (if data extends beyond test window)
    """
    # ── Resolve FUTURE_STEPS ────────────────────────────────────────────
    if FUTURE_STEPS is None:
        FUTURE_STEPS = N_STEPS
    
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df_before_targets = feature_engineering(data)
    # Build targets for alignment purposes (to know the test window boundaries)
    df_with_targets = build_targets(df_before_targets, N_STEPS)

    features, labels = get_features_and_labels(df_with_targets, TARGET_COLS)
    
    # Determine test window boundaries
    n = len(df_with_targets)
    train_end = int(n * DATA_SPLIT_RATIOS[0])
    val_end = train_end + int(n * DATA_SPLIT_RATIOS[1])
    test_end = min(val_end + int(n * DATA_SPLIT_RATIOS[2]), n)
    
    # ── Extract actual test window data ──────────────────────────────────
    # Use the actual observed prices, not the shifted/computed targets
    test_start_idx = val_end
    test_window_df = df_before_targets.iloc[test_start_idx:test_end].copy()
    
    if len(test_window_df) == 0:
        raise ValueError("No test data available after train/val split")
    
    # Build actual observed OHLC returns relative to opening price
    actual_rel = np.column_stack([
        np.ones(len(test_window_df)),                           # open_rel = 1.0 (by definition)
        test_window_df["high"].values / test_window_df["open"].values,    # high_rel
        test_window_df["low"].values / test_window_df["open"].values,     # low_rel
        test_window_df["close"].values / test_window_df["open"].values,   # close_rel
    ])
    
    actual_opens = test_window_df["open"].values
    
    # Generate dates for test window (target dates, shifted by N_STEPS)
    dates = generate_trading_dates(
        test_window_df.index[0],
        len(test_window_df),
        N_STEPS
    )
    seed_open = df_before_targets["open"].values[test_start_idx - 1]
    
    # Build result dataframe using actual prices
    result_df = build_result_df(actual_rel, actual_opens, idx=dates, 
                                seed_open=seed_open, n_steps=N_STEPS)
    
    # ── Predict future rows if data extends beyond test window ────────────
    n_future = len(df_before_targets) - test_end
    if n_future > 0 and FUTURE_STEPS > 0:
        future_df_raw = df_before_targets.iloc[test_end:].copy()
        
        # Build actual observed OHLC returns for future rows
        future_rel = np.column_stack([
            np.ones(len(future_df_raw)),
            future_df_raw["high"].values / future_df_raw["open"].values,
            future_df_raw["low"].values / future_df_raw["open"].values,
            future_df_raw["close"].values / future_df_raw["open"].values,
        ])
        
        future_opens = future_df_raw["open"].values
        
        # Generate future target dates
        last_test_date = result_df.index[-1]
        future_dates = generate_future_dates(last_test_date, len(future_df_raw), N_STEPS)
        
        # Use the last daisy-chained close from test as continuation point
        continuation_close = result_df["daisy_chained_close"].iloc[-1]
        
        future_df = build_result_df(future_rel, future_opens, idx=future_dates,
                                    seed_open=seed_open, n_steps=N_STEPS,
                                    continuation_close=continuation_close)
        result_df = pd.concat([result_df, future_df], ignore_index=False)
    
    if return_metrics:
        # For Seer, RMSE is meaningless (it's perfect by definition)
        rmse_scores = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "mean": 0.0}
        return (result_df, rmse_scores)
    
    return result_df