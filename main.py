import importlib
import os
import numpy as np
import pandas as pd

try:
    from pandas_market_calendars import get_calendar
    HAS_MARKET_CAL = True
except ImportError:
    HAS_MARKET_CAL = False
    print("Warning: pandas_market_calendars not installed. Using weekday count instead.")
    print("Install with: pip install pandas_market_calendars")

from upgraded_utilities import save_models_data, plot_candlestick_graphs, plot_mp_graphs

from backtest import main as bt

from model_evaluator import main as me


# Hyperparemeter Tuning toggle
RUN_TUNING = False

# Which tuning method(s) to run - Can be any subset of: "grid", "random", "bayesian"
TUNING_METHOD = "random"

# Which models to tune — None tunes all models in TUNABLE_MODELS.
# If only some need tuning, provide a list such as ["Modularised_ANN", "Modularised_DTR"]
TUNING_MODELS = None

# Shared tuning parameters — used by all three tuning methods
TUNING_KWARGS = dict(
    start_date = "2010-01-01",
    end_date = "2025-12-31",
    data_split_ratios = (0.8, 0.1, 0.1),
    n_steps = 5,
    rmse_mode = "price",
)

# Method-specific overrides (merged with TUNING_KWARGS when tuning is turned on).

# No additional kwargs needed
GRID_KWARGS    = {}
# Keep a fixed seed for consistency and determinism
RANDOM_KWARGS  = dict(n_trials=20, seed=42)
BAYESIAN_KWARGS = dict(n_trials=25, n_initial=5)


# --- Configure prediction horizon ---
# Set your target end date for predictions
END_DATE = "2027-04-18"

# Load just the data dates to find the last one
df_dates = pd.read_csv("Data/S&P 500 Composite.csv", usecols=["YYYYMMDD"])
df_dates["Date"] = pd.to_datetime(df_dates["YYYYMMDD"], format="%Y%m%d")
last_data_date = df_dates["Date"].max()
last_data_date_str = last_data_date.strftime("%Y-%m-%d")

# Calculate N_STEPS using actual NYSE trading calendar
if HAS_MARKET_CAL:
    try:
        nyse = get_calendar('NYSE')
        # Get all trading days between last data date and END_DATE
        trading_days = nyse.sessions_window(last_data_date, END_DATE)
        # N_STEPS is the count of trading days (excluding the start date)
        N_STEPS = max(1, len(trading_days) - 1)
        print(f"Using NYSE calendar: {N_STEPS} trading days from {last_data_date_str} to {END_DATE}")
    except Exception as e:
        print(f"Error using NYSE calendar: {e}")
        print("Falling back to weekday count...")
        start = np.datetime64(last_data_date_str, 'D')
        end = np.datetime64(END_DATE, 'D')
        N_STEPS = max(1, int(np.busday_count(start, end)))
        print(f"Predicting {N_STEPS} trading days (weekday estimate) to {END_DATE}")
else:
    # Fallback to weekday count if market calendars not available
    start = np.datetime64(last_data_date_str, 'D')
    end = np.datetime64(END_DATE, 'D')
    N_STEPS = max(1, int(np.busday_count(start, end)))
    print(f"Predicting {N_STEPS} trading days (weekday estimate) to {END_DATE}")

print(f"Last data date: {last_data_date_str}")
# --- End configuration ---


if RUN_TUNING:
    if TUNING_METHOD == "grid":
        from Tuning.run_grid_search import run_all as _tune
        _tune(models=TUNING_MODELS, **TUNING_KWARGS, **GRID_KWARGS)

    elif TUNING_METHOD == "random":
        from Tuning.run_random_search import run_all as _tune
        _tune(models=TUNING_MODELS, **TUNING_KWARGS, **RANDOM_KWARGS)

    elif TUNING_METHOD == "bayesian":
        from Tuning.run_bayesian_opt import run_all as _tune
        _tune(models=TUNING_MODELS, **TUNING_KWARGS, **BAYESIAN_KWARGS)

    else:
        raise ValueError(f"Unknown TUNING_METHOD={TUNING_METHOD!r}. Please choose from 'grid', 'random', or 'bayesian'.")


# Now evaluate the models

# All the model files are in a subdirectory called Models, all end in .py and any files to not be tested should start with _
files = [f[:-3] for f in os.listdir("./Models") if f.endswith(".py") and not f.startswith("_")]

# Stores the main function for each model as the value, with the key being the name of the file
models = {}
# Stores a pandas dataframe outputted by the main function from each model as the value, with the key being the name of the model's file
model_results = {}

# Try each model, if it has a 'main' function, execute it and record the results
for name in files:
    module = importlib.import_module(f"Models.{name}")
    if hasattr(module, "main"):
        models[name] = module.main
    else:
        print(f"{name} has no main() — skipping")
"""
# Iterates through every model, retrieving the name and main function
for name, main_function in models.items():
    try:
        # Each model returns a DataFrame with columns:
        # open_rel, high_rel, low_rel, close_rel, open, high, low, close
        # And the daisy chained versions for each, where the relatives (returns) should be identical to before,
        # but are displayed as a sanity check - if unequal there's an error
        model_results[name] = main_function(END_DATE=END_DATE, N_STEPS=N_STEPS)
    except Exception as e:
        print(f"{name} failed: {e}")
"""

# Iterates through every model, retrieving the name and main function
for name, main_function in models.items():
    try:
        # Each model returns a DataFrame with columns:
        # open_rel, high_rel, low_rel, close_rel, open, high, low, close
        # And the daisy chained versions for each, where the relatives (returns) should be identical to before,
        # but are displayed as a sanity check - if unequal there's an error
        model_results[name] = main_function(
            END_DATE=END_DATE,
            # Always 1-step for train/test
            N_STEPS=1,
            # The large computed horizon for future predictions
            FUTURE_STEPS=N_STEPS
        )
    except Exception as e:
        print(f"{name} failed: {e}")


# --- Save all the models data into CSV files ---
for name in models.keys():
    try:
        save_models_data(name, model_results[name])
    except Exception as e:
        print(f"Failed saving {name} - with error {e}")


# Display the results in a candlestick plot and save the matplotlib plots
for name, df in model_results.items():
    # Call the plot matplot lib graphs function to render and save those plots
    plot_mp_graphs(df, name, save=True, show=False)
    # And call the pyplot candlesticks function to render, save and display those results
    plot_candlestick_graphs(df, name, save=True, show=False)


# Now run the backtest!
bt(save=True, show=True)

# And get metrics for each file!
me()