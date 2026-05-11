import pandas as pd
import numpy as np
import os

def get_models_data(model_path):
    df = pd.read_csv(model_path, index_col = 0, header = 0)
    return df

def parse_dates_safe(idx):
    try:
        dt_idx = pd.to_datetime(idx, dayfirst = True)
        if dt_idx.isna().all():
            dt_idx = pd.to_datetime(idx, dayfirst = False)
    except:
        dt_idx = pd.to_datetime(idx, dayfirst = False)
    return dt_idx



def main():
    required_cols = ["open", "high", "low", "close"]

    # Load Seer
    seer = get_models_data("Newest Results/Seer.csv")
    seer = seer[required_cols].copy()
    seer.index = parse_dates_safe(seer.index)
    seer = seer[~seer.index.duplicated(keep = 'first')]
    seer[required_cols] = seer[required_cols].apply(pd.to_numeric, errors = 'coerce')
    seer = seer.sort_index()

    # Find all model CSVs
    files = [f for f in os.listdir("./Newest Results") if f.endswith(".csv") and not f.startswith("_")]

    results = []

    for f in files:
        if f == "Seer.csv":
            continue

        df = get_models_data(f"Newest Results/{f}")

        # Keep only required columns
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Skipping {f}, missing columns: {missing}")
            continue

        df = df[required_cols].copy()
        df.index = parse_dates_safe(df.index)
        df = df[~df.index.duplicated(keep = 'first')]
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors = 'coerce')
        df = df.sort_index()

        # Allign using forward fill - needed for weekly adjusteds
        df_aligned = df.copy()
        seer_aligned = seer.reindex(df.index, method = 'ffill')  # forward fill

        
        mae = (df_aligned - seer_aligned).abs()
        mse = ((df_aligned - seer_aligned) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - ((seer_aligned - df_aligned) ** 2).sum() / ((seer_aligned - seer_aligned.mean()) ** 2).sum()
        dir_acc = (np.sign(seer_aligned.diff()) == np.sign(df_aligned.diff())).astype(int)

        # Flatten all metrics for CSV
        metrics_flat = {}
        for col in required_cols:
            metrics_flat[f"mae_{col}"] = mae[col].mean()
            metrics_flat[f"mse_{col}"] = mse[col].mean()
            metrics_flat[f"rmse_{col}"] = rmse[col].mean()
            metrics_flat[f"r2_{col}"] = r2[col]
            metrics_flat[f"dir_acc_{col}"] = dir_acc[col].mean()

        metrics_flat["model"] = os.path.splitext(f)[0]

        results.append(metrics_flat)

        print("\n"*2)
        print(metrics_flat["model"])
        print(metrics_flat)

    # Save results
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.set_index("model")
    output_path = "Newest Results/Testing Results/Metrics.csv"
    metrics_df.to_csv(output_path)
    
