import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from upgraded_utilities import feature_engineering, walk_forward_validation, get_data, get_features_and_labels, predict_future_rows
from model_utils import build_targets, build_result_df, compute_rmse, build_result_df_chained, TARGET_COLS, generate_trading_dates, generate_future_dates


class ANNModel(nn.Module):

    def __init__(self, input_size, width = 256, dropout = 0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, width),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 2, width // 4),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 4, width // 8),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 8, max(width // 16, 8)), nn.ReLU(),
            nn.Linear(max(width // 16, 8), 4),
        )

    def forward(self, x):
        return self.model(x)


def make_ann_predict(epochs = 50, lr = 0.001, width = 256, dropout = 0.2, batch_size = 32):
    # Added parameters for fine tuning
    def ann_predict(feature_train_data, label_train_data, feature_test_data, k = None):
        # feature data is already standardised by walk_forward_validation
        feature_train = torch.tensor(feature_train_data, dtype = torch.float32)
        label_train = torch.tensor(label_train_data,   dtype = torch.float32)
        feature_test = torch.tensor(feature_test_data,  dtype = torch.float32)

        # Define the model with constraints defined above
        model = ANNModel(feature_train_data.shape[1], width = width, dropout = dropout)
        optimiser = optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
        loss_function = nn.MSELoss()
        loader = DataLoader(TensorDataset(feature_train, label_train), batch_size = batch_size, shuffle = True)

        # Train the model
        model.train()
        for _ in range(epochs):
            for features_b, labels_b in loader:
                loss = loss_function(model(features_b), labels_b)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        # Evaluate the model and return the predictions
        model.eval()
        with torch.no_grad():
            return model(feature_test).numpy()   # (n, 4)

    return ann_predict


def main(START_DATE = "2010-01-01", END_DATE = "2025-12-31",
        DATA_SPLIT_RATIOS = (0.8, 0.1, 0.1),
        N_STEPS = 1,
        rmse_mode = "price",
        # Tunable parameters
        epochs = 50,
        lr = 0.001,
        width = 256,
        dropout = 0.2,
        batch_size = 32,
        # Flag for tuning evaluation
        FUTURE_STEPS = None,
         return_metrics = False):
    # If return metrics is true, returns (Dataframe, rmse dictionary) instead of just the dataframe
    # Will be used to retrieve scores without re-running or recomputing errors

    # Get the data and build the appropriate dataframes
    if FUTURE_STEPS is None:
        FUTURE_STEPS = N_STEPS
    
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    data_before_targets = feature_engineering(data)
    data = build_targets(data_before_targets, N_STEPS)

    # Sanity check - ensure there is data to be used
    if len(data) == 0:
        raise ValueError(f"No data remaining after feature engineering for {START_DATE}-{END_DATE}.")

    # Split data into features and labels
    features, labels = get_features_and_labels(data, TARGET_COLS)

    # Get models predictions, ignored value is RMSE
    predicted, actual, _ = walk_forward_validation(
        features, labels,
        make_ann_predict(epochs = epochs, lr = lr, width = width, dropout = dropout, batch_size = batch_size),
        data_split_ratios = DATA_SPLIT_RATIOS,
    )

    # If there are no predictions, throw an error
    if len(predicted) == 0:
        raise ValueError("walk_forward_validation produced no predictions.")

    # Find the end of the training window to find the first value from testing
    training_window = max(1, int(len(features) * DATA_SPLIT_RATIOS[0]))
    # Find the actual opens after the training data, to the end of the dataset
    # NOTE - assumes no validation
    actual_opens = data["open"].values[training_window: training_window + len(predicted)]
    # Retrieve the dates of the data to recompile the dataframe
    # Generate continuous trading dates for predictions (eliminates date gaps)
    dates = generate_trading_dates(
        data["Date"].values[training_window],
        len(predicted),
        N_STEPS
    )
    # Gets the 'seeds' used for reconstructing true n-step ahead predictions
    seed_open = data["open"].values[training_window - 1]

    # Gets the RMSE scores for all values (OHLC + mean) for testing purposes
    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode = rmse_mode)
    # Reports the RMSE
    print(f"ANN ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    # Build and return the dataframe
    result_df = build_result_df(predicted, actual_opens, idx = dates, seed_open = seed_open, n_steps = N_STEPS)

    # Predict future rows if data extends beyond training window
    feature_cols = [c for c in data_before_targets.select_dtypes(include = [np.number]).columns
                     if c not in TARGET_COLS]
    features_full = data_before_targets[feature_cols].values
    future_preds, future_opens, future_dates = predict_future_rows(
        data_before_targets, data, features_full, features, labels, 
        make_ann_predict(epochs = epochs, lr = lr, width = width, dropout = dropout, batch_size = batch_size),
        N_STEPS,
    )
    if future_preds is not None:
        # Generate future target dates starting from where test ended (no overlap)
        last_test_date = result_df.index[-1]
        future_target_dates = generate_future_dates(last_test_date, len(future_preds), N_STEPS)
        
        # Use the last daisy-chained close from test predictions as continuation point
        # This ensures daisy-chained prices connect seamlessly without jumping
        continuation_close = result_df["daisy_chained_close"].iloc[-1]
        
        future_df = build_result_df(future_preds,
                                    future_opens,
                                    idx = future_target_dates,
                                    seed_open = seed_open, n_steps = N_STEPS,
                                    continuation_close = continuation_close)
        result_df = pd.concat([result_df, future_df], ignore_index = False)

    if return_metrics:
        return (result_df, rmse_scores)
    
    else:
        return result_df

