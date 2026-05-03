import numpy as np
import pandas as pd

# Global constants - used for yfinance values and keeping WRDS models consistent
COLS = ["open", "high", "low", "close"]
# Column names for target construction — same order as output neurons
TARGET_COLS = ["target_open", "target_high", "target_low", "target_close"]


def generate_trading_dates(start_date, n_predictions, n_steps=1):
    """
    Generate continuous trading dates for predictions.
    
    Parameters
    ----------
    start_date : pandas.Timestamp or str
        The starting REFERENCE date (when first prediction was made)
    n_predictions : int
        Number of predictions to generate dates for
    n_steps : int
        Prediction horizon (how many days ahead each prediction is)
    
    Returns
    -------
    pd.DatetimeIndex
        Trading dates for each prediction's TARGET date (reference + n_steps trading days)
        
    Note: These are TARGET dates (what we're predicting for), not reference dates.
          Reference dates are assumed to be continuous trading days starting from start_date.
    """
    try:
        from pandas_market_calendars import get_calendar
        nyse = get_calendar('NYSE')
        
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        # Get a sufficient range of trading days
        end_date_estimate = start_date + pd.Timedelta(days=365 * 3)
        all_trading_days = nyse.sessions_window(start_date, end_date_estimate)
        
        # Find the index of start_date in the trading days
        try:
            start_idx = list(all_trading_days).index(pd.Timestamp(start_date))
        except ValueError:
            # If start_date is not in trading days (weekend/holiday), find next trading day
            start_idx = 0
            for i, td in enumerate(all_trading_days):
                if td >= start_date:
                    start_idx = i
                    break
        
        # For each prediction, the reference is at start_idx + i, target is at start_idx + i + n_steps
        target_dates = []
        for i in range(n_predictions):
            ref_idx = start_idx + i
            target_idx = ref_idx + n_steps
            
            if target_idx < len(all_trading_days):
                target_dates.append(all_trading_days[target_idx])
            else:
                # Fallback if we run out of trading days
                last_date = all_trading_days[-1]
                days_beyond = target_idx - len(all_trading_days)
                target_dates.append(last_date + pd.tseries.offsets.BDay(days_beyond))
        
        return pd.DatetimeIndex(target_dates)
    
    except (ImportError, Exception) as e:
        # Fallback to business day offset
        start_date = pd.Timestamp(start_date)
        return pd.bdate_range(start=start_date, periods=n_predictions, freq='B') + pd.tseries.offsets.BDay(n_steps)

def generate_future_dates(last_test_date, n_future_predictions, n_steps=1):
    """
    Generate target dates for future predictions, continuing from test predictions.
    
    Parameters
    ----------
    last_test_date : pandas.Timestamp or str
        The LAST target date from the test window (not reference date)
    n_future_predictions : int
        Number of future predictions to generate dates for
    n_steps : int
        Prediction horizon
    
    Returns
    -------
    pd.DatetimeIndex
        Target dates for future predictions, starting right after last_test_date
    """
    try:
        from pandas_market_calendars import get_calendar
        nyse = get_calendar('NYSE')
        
        if isinstance(last_test_date, str):
            last_test_date = pd.Timestamp(last_test_date)
        
        # Get trading days starting from the day after last_test_date
        end_date_estimate = last_test_date + pd.Timedelta(days=365)
        all_trading_days = nyse.sessions_window(last_test_date, end_date_estimate)
        
        # Find last_test_date in the calendar
        try:
            last_idx = list(all_trading_days).index(pd.Timestamp(last_test_date))
        except ValueError:
            # If not found exactly, find the closest date
            last_idx = 0
            for i, td in enumerate(all_trading_days):
                if td >= last_test_date:
                    last_idx = i
                    break
        
        # Future dates start right after last_test_date
        future_dates = []
        for i in range(n_future_predictions):
            idx = last_idx + 1 + i
            if idx < len(all_trading_days):
                future_dates.append(all_trading_days[idx])
            else:
                # Fallback
                last_date = all_trading_days[-1]
                future_dates.append(last_date + pd.tseries.offsets.BDay(idx - len(all_trading_days) + 1))
        
        return pd.DatetimeIndex(future_dates)
    
    except (ImportError, Exception):
        # Fallback to business day offset
        last_test_date = pd.Timestamp(last_test_date)
        return pd.bdate_range(start=last_test_date + pd.tseries.offsets.BDay(1), 
                             periods=n_future_predictions, freq='B')

# If n_steps is 0, the labels would be the current data, so n should always be larger
# than 0 (1 or more) to ensure the target labels are 'future'
def build_targets(df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    # Calculate the OHLC returns for labels 
    df = df.copy()
    df["target_open"] = df["open"].shift(-n_steps) / df["open"]
    df["target_high"] = df["high"].shift(-n_steps) / df["open"]
    df["target_low"] = df["low"].shift(-n_steps) / df["open"]
    df["target_close"] = df["close"].shift(-n_steps) / df["open"]
    # Drop any NaNs
    return df.dropna()

# The daisy chained results data frame function, used for true n step ahead predictions
def build_result_df_chained(predicted_rel: np.ndarray,
                             seed_open: float,
                             n_steps: int = 1,
                             idx=None) -> pd.DataFrame:
    predicted_rel  = np.asarray(predicted_rel)
    n              = len(predicted_rel)

    # ── 1. Build chain on non-overlapping anchor points ───────────────────────
    chain_indices  = np.arange(0, n, max(n_steps, 1))        # [0, 5, 10, ...]
    chain_open_rel = predicted_rel[chain_indices, 0]          # (m,)
    m              = len(chain_open_rel)
    chain_opens    = np.empty(m)
    current_open   = seed_open
    for i in range(m):
        chain_opens[i] = chain_open_rel[i] * current_open
        current_open   = chain_opens[i]                       # feeds next link

    # ── 2. Interpolate chained open back to every step ────────────────────────
    full_open = np.interp(np.arange(n), chain_indices, chain_opens)   # (n,)

    # ── 3. Apply intraday ratios from raw model predictions ───────────────────
    safe_open_rel  = np.where(np.abs(predicted_rel[:, 0]) < 1e-8, 1.0, predicted_rel[:, 0])
    intraday_high  = predicted_rel[:, 1] / safe_open_rel
    intraday_low   = predicted_rel[:, 2] / safe_open_rel
    intraday_close = predicted_rel[:, 3] / safe_open_rel

    prices = np.column_stack([
        full_open,
        full_open * intraday_high,
        full_open * intraday_low,
        full_open * intraday_close,
    ])

    if idx is None:
        idx = pd.RangeIndex(n)

    return pd.DataFrame({
        "open_rel":  predicted_rel[:, 0],
        "high_rel":  predicted_rel[:, 1],
        "low_rel":   predicted_rel[:, 2],
        "close_rel": predicted_rel[:, 3],
        "open":      prices[:, 0],
        "high":      prices[:, 1],
        "low":       prices[:, 2],
        "close":     prices[:, 3],
    }, index=idx)


def build_result_df(predicted_rel: np.ndarray,
                    actual_opens: np.ndarray,
                    idx=None,
                    seed_open: float = None,
                    n_steps: int = 1,
                    continuation_close: float = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    predicted_rel : (n, 4)  [open_rel, high_rel, low_rel, close_rel]
    actual_opens  : (n,)    open price at each row's reference time t.
                            Reconstructed price[t] = predicted_rel[t] * actual_opens[t].
    idx           : array-like of timestamps, or None (falls back to RangeIndex)
    seed_open     : float   The last known open price from the training window
                            (i.e. df["open"].values[training_window - 1]).
                            Used as the starting point for daisy-chained predictions
                            so no test-set data is touched. If None, falls back to
                            actual_opens[0] (old behaviour — uses test data).
    n_steps       : int     Prediction horizon — passed to build_result_df_chained
                            so the chain subsamples correctly (see that function).
    continuation_close : float
                        The closing price from the previous segment (test window).
                        For future predictions, this should be the last daisy-chained
                        close from the test window to ensure continuity. If provided,
                        overrides seed_open for daisy-chain initialization.

    Returns
    -------
    pd.DataFrame with columns:
        open_rel, high_rel, low_rel, close_rel, open, high, low, close
    
    Note: idx should already be target dates (shifted forward by n_steps).
          This function does NOT shift them further.
    """
    predicted_rel = np.asarray(predicted_rel)
    actual_opens = np.asarray(actual_opens).ravel()
    
    # Use continuation_close if provided (for seamless future prediction continuity)
    if continuation_close is not None:
        effective_seed = continuation_close
    elif seed_open is None:
        effective_seed = float(actual_opens[0])   # fallback — uses test data
    else:
        effective_seed = seed_open
    
    actual_opens  = np.asarray(actual_opens).reshape(-1, 1)  # (n, 1) for broadcasting

    if idx is None:
        idx = pd.RangeIndex(len(predicted_rel))

    prices = predicted_rel * actual_opens   # (n, 4) — each row scaled by its own open

    daisy_chained_df = build_result_df_chained(predicted_rel, seed_open=effective_seed, n_steps=n_steps)

    return pd.DataFrame({
        "open_rel":  predicted_rel[:, 0],
        "high_rel":  predicted_rel[:, 1],
        "low_rel":   predicted_rel[:, 2],
        "close_rel": predicted_rel[:, 3],
        "open":      prices[:, 0],
        "high":      prices[:, 1],
        "low":       prices[:, 2],
        "close":     prices[:, 3],
        "daisy_chained_open_rel":  daisy_chained_df["open_rel"].values,
        "daisy_chained_high_rel":  daisy_chained_df["high_rel"].values,
        "daisy_chained_low_rel":   daisy_chained_df["low_rel"].values,
        "daisy_chained_close_rel": daisy_chained_df["close_rel"].values,
        "daisy_chained_open":      daisy_chained_df["open"].values,
        "daisy_chained_high":      daisy_chained_df["high"].values,
        "daisy_chained_low":       daisy_chained_df["low"].values,
        "daisy_chained_close":     daisy_chained_df["close"].values,
    }, index=idx)

# Computes the RMSE between predicted and actual values
# Inputs:
#   predicted_rel - models relative predictions (returns) with shape (n,4)
#   actual_red - the actual relative predictions (returns) with shape (n,4)
#   actual_opens - the opening price at each step - used to convert returns into prices
#   mode - Either 'price' or 'relative' - computes the RMSE on the prices or the returns respectively
# Outputs:
#   Dictionary containing the RMSE per column and the mean RMSE of the OHLC RMSEs
#    Should output 5 values per column - OHLC + mean
def compute_rmse(predicted_rel: np.ndarray,
                 actual_rel: np.ndarray,
                 actual_opens: np.ndarray,
                 mode: str = "price") -> dict:
    # Ensure the actual opens are of correct shape (inputs shape (n,1))
    actual_opens = np.asarray(actual_opens).reshape(-1, 1)

    # Case 1 - evaluate the prices
    if mode == "price":
        pred  = predicted_rel * actual_opens
        truth = actual_rel    * actual_opens
    # Case 2 - evaluate the returns
    elif mode == "relative":
        pred  = predicted_rel
        truth = actual_rel
    # Case 3 - invalid option, throw an error
    else:
        raise ValueError(f"rmse_mode must be 'price' or 'relative', got '{mode}'")

    # Calculate the mean RMSE for each column (OHLC)
    per_col = {col: float(np.sqrt(np.mean((pred[:, i] - truth[:, i]) ** 2)))
               for i, col in enumerate(COLS)}
    
    # Calculate the mean RMSE across those 4 OHLC errors
    per_col["mean"] = float(np.mean(list(per_col.values())))
    return per_col
