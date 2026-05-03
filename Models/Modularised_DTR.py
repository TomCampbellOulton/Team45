import numpy as np
import pandas as pd

from upgraded_utilities import (feature_engineering, get_data, get_features_and_labels,
                                 get_feature_names, dtr_walk_forward_validation, predict_future_rows)
from model_utils import build_targets, build_result_df, compute_rmse, generate_future_dates, TARGET_COLS, generate_trading_dates


class Node:
    def __init__(self, feature_index=None, threshold=None,
                 left_child=None, right_child=None, prediction=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.prediction = prediction


class DecisionTreeRegressor:
    def __init__(self, max_depth=2, min_samples_split=1, min_samples_leaf=5,
                 min_impurity_decrease=1e-6, min_variance=1e-8, max_features=None):
        self.max_depth = max_depth
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_variance = min_variance
        self.feature_names = None
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _mse(self, labels):
        if len(labels) == 0:
            return 0.0
        return float(np.mean((labels - np.mean(labels, axis=0)) ** 2))

    def _dataset_split(self, features_data, labels, feature_index, threshold):
        left_mask  = features_data[:, feature_index] <= threshold
        right_mask = ~left_mask
        return (features_data[left_mask],  labels[left_mask],
                features_data[right_mask], labels[right_mask])

    def _find_optimal_split(self, features_data, labels, current_depth):
        best_feature_index = None
        best_threshold = None
        lowest_error = float("inf")
        parent_error = self._mse(labels)

        feature_indices = np.arange(features_data.shape[1])
        if self.max_features is not None:
            feature_indices = np.random.choice(
                feature_indices, min(self.max_features, len(feature_indices)), replace=False)

        for fi in feature_indices:
            unique_vals = np.unique(features_data[:, fi])
            thresholds  = (unique_vals[:-1] + unique_vals[1:]) / 2
            for threshold in thresholds:
                _, ll, _, rl = self._dataset_split(features_data, labels, fi, threshold)
                if len(ll) < self.min_samples_leaf or len(rl) < self.min_samples_leaf:
                    continue
                error = (len(ll) * self._mse(ll) + len(rl) * self._mse(rl)) / len(labels)
                gain  = (parent_error - error) / (1 + current_depth)
                if gain < self.min_impurity_decrease or error >= lowest_error:
                    continue
                lowest_error, best_feature_index, best_threshold = error, fi, threshold

        return best_feature_index, best_threshold, lowest_error

    def _build_tree(self, features_data, labels, current_depth):
        if (current_depth >= self.max_depth
                or len(labels) < self.min_samples_split
                or np.var(labels) < self.min_variance):
            return Node(prediction=np.mean(labels, axis=0))

        best_fi, best_thresh, lowest_error = self._find_optimal_split(
            features_data, labels, current_depth)

        if best_fi is None or (self._mse(labels) - lowest_error) < self.min_impurity_decrease:
            return Node(prediction=np.median(labels, axis=0))

        lf, ll, rf, rl = self._dataset_split(features_data, labels, best_fi, best_thresh)
        return Node(
            feature_index=best_fi, threshold=best_thresh,
            left_child=self._build_tree(lf, ll, current_depth + 1),
            right_child=self._build_tree(rf, rl, current_depth + 1),
        )

    def fit(self, features_data: np.ndarray, labels: np.ndarray, feature_names=None):
        self.feature_names = feature_names
        self.root          = self._build_tree(features_data, labels, 0)

    def _predict_single(self, node, feature_row):
        if node.prediction is not None:
            return node.prediction
        if feature_row[node.feature_index] <= node.threshold:
            return self._predict_single(node.left_child, feature_row)
        return self._predict_single(node.right_child, feature_row)

    def predict(self, features_data: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(self.root, row) for row in features_data])


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         N_STEPS=1,
         rmse_mode="price",
         # ── Tunable hyperparameters ──────────────────────────────────────
         max_depth=1,
         min_samples_split=5,
         min_samples_leaf=15,
         min_impurity_decrease=1e-6,
         max_features=None,
         # ── Evaluation flag ─────────────────────────────────────────────
         FUTURE_STEPS=None,
         return_metrics=False):
    # ── Resolve FUTURE_STEPS ────────────────────────────────────────────
    if FUTURE_STEPS is None:
        FUTURE_STEPS = N_STEPS
    
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df_before_targets = feature_engineering(data)
    df   = build_targets(df_before_targets, N_STEPS)

    feature_names = get_feature_names(df, TARGET_COLS)
    features, labels = get_features_and_labels(df, TARGET_COLS)

    predicted, actual, _, test_indices = dtr_walk_forward_validation(
        features, labels, DecisionTreeRegressor,
        data_split_ratios=DATA_SPLIT_RATIOS,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        max_features=max_features,
        k=None,
    )

    actual_opens = df["open"].values[test_indices]
    # Generate continuous trading dates for predictions (eliminates date gaps)
    dates = generate_trading_dates(
        df.index[test_indices[0]],
        len(predicted),
        N_STEPS
    )
    seed_open    = df["open"].values[test_indices[0] - 1]

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"DTR ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    result_df = build_result_df(predicted, actual_opens, idx=dates,
                                seed_open=seed_open, n_steps=N_STEPS)
    
    last_test_date = END_DATE  # Last date from test window predictions
    
    # Check if df_before_targets has rows after last_test_date
    future_mask = df_before_targets.index > last_test_date
    
    if future_mask.any():
        # If theres data after the testing window, predict on it anyway c:
        future_start_idx = np.where(future_mask)[0][0]
        
        feature_cols  = [c for c in df_before_targets.select_dtypes(include=[np.number]).columns
                         if c not in TARGET_COLS]
        features_full = df_before_targets[feature_cols].values
        
        mean = features.mean(axis=0)
        std  = features.std(axis=0) + 1e-8
        
        # Train on all labeled data
        future_dtr = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
        )
        future_dtr.fit((features - mean) / std, labels)
        
        # Only make predictions on dates after the end of testing
        future_preds = future_dtr.predict((features_full[future_start_idx:] - mean) / std)
 
        # Use the last daisy-chained close from test predictions as continuation point
        continuation_close = result_df["daisy_chained_close"].iloc[-1]
        
        #Only use the future rows for building the df o:
        future_df = build_result_df(future_preds,
                                    df_before_targets["open"].values[future_start_idx:],
                                    idx=df_before_targets.index[future_start_idx:],
                                    seed_open=seed_open, n_steps=N_STEPS,
                                    continuation_close=continuation_close)
        result_df = pd.concat([result_df, future_df], ignore_index=False)
    
    # Shouldn't be needed but is for some reason - remove any duplicates o:
    if result_df.index.duplicated().any():
        n_dupes = result_df.index.duplicated().sum()
        print(f"Removing {n_dupes} duplicate date(s) in DTR")
        result_df = result_df[~result_df.index.duplicated(keep='first')]
    
    return (result_df, rmse_scores) if return_metrics else result_df
