import pandas as pd
import numpy as np
# Only used for saving model output with timestamps
from datetime import datetime
# Plotting predicted prices and returns
import matplotlib.pyplot as plt
# For plotting the candlesticks, found in - https://plotly.com/python/candlestick-charts/
# IMPORTANT NOTE: For plotly's save to image function 'write_image()', kaleido must be installed...
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Function for getting the data from yfinance - mainly for testing
def get_data_yf(start_date, end_date, ticker = "^GSPC"):
    import yfinance as yf
    df = yf.download(ticker, start = start_date, end = end_date)
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace = True)
    return df

# Gets the data from the csv file from WRDS and combines with yfinance data
# using yfinance's ratios for OHLC to multiply WRDS' daily value by (WRDS is more reliable)
def get_data(start_date, end_date, data_file = "Data/S&P 500 Composite.csv"):
    # Reads only relevant columns from WRDS data - date, daily return and daily price
    df = pd.read_csv(data_file, usecols = ["YYYYMMDD", "DlyPrcRet", "DlyPrcInd"])

    # Convert the string date to pandas datetime
    df["Date"] = pd.to_datetime(df["YYYYMMDD"], format = "%Y%m%d")
    # Use datetime as the index for dataframe
    df.set_index("Date", inplace = True)

    # Get the start and end date in pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data to only be between the start and end date inclusively
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    # Sort the data by date
    filtered_df.sort_index(inplace = True)

    # Fetch OHLC ratios from yfinance
    y_finance = get_data_yf(start_date, end_date)
    y_finance = y_finance.reindex(filtered_df.index)

    # Relative effectively being return, NOTE - open not used as WRDS
    # is being used as the 'open' prices - yfinance's raw prices not used, only
    # their daily ratios relative to market open
    relative_high = y_finance["high"]  / y_finance["open"]
    relative_low = y_finance["low"]   / y_finance["open"]
    relative_close = y_finance["close"] / y_finance["open"]

    # Use WRDS data as the open price for the day
    base_price = filtered_df["DlyPrcInd"]
    # Then use YFinance to fill in the high, low, close - not using yfinances exact values
    # but rather using their ratios for high, low, close relative to open and then applying
    # that ratio to the open (WRDS data)
    filtered_df["open"] = base_price
    filtered_df["high"] = base_price * relative_high
    filtered_df["low"] = base_price * relative_low
    filtered_df["close"] = base_price * relative_close

    # Drop non-numeric columns (YYYYMMDD, etc.)
    filtered_df = filtered_df.drop(columns = ["YYYYMMDD"], errors = "ignore")
    filtered_df = filtered_df.select_dtypes(include = ["number"])

    return filtered_df


# Input: df - pandas dataframe containing the feature-engineered columns AND the target columns list
#   target_cols - list of strings, columns to be used as labels (excluded from features)
# Output: 2 numpy arrays, one for features and the other containing the matching labels
#   features - np.ndarray of size (n, n_features)
#   labels - np.ndarray of size (n, len(target_cols))
def get_features_and_labels(df: pd.DataFrame, target_cols: list):
    # Extract only numeric data from the inputted dataframe
    df_numeric = df.select_dtypes(include = [np.number]).copy()
    # Any numeric columns remaining are kept if they're not in the target_cols (to be the labels)
    feature_cols = [c for c in df_numeric.columns if c not in target_cols]
    features = df_numeric[feature_cols].values
    labels = df_numeric[target_cols].values

    return features, labels

# Simple return for extracting a list of the feature names from the dataframe
def get_feature_names(df: pd.DataFrame, target_cols: list) -> list:
    # Returns the feature column names - includes everything that's numeric EXCEPT any target columns
    df_numeric = df.select_dtypes(include = [np.number])
    return [c for c in df_numeric.columns if c not in target_cols]


# Builds pattern dataset for KNN with Pattern Matching
# Takes an array of the returns and a window,
# Outputs features - rolling window of past returns, labels - next return after the window
def build_pattern_dataset(returns, window = 20):
    features, labels = [], []
    for i in range(window, len(returns) - 1):
        features.append(returns[i - window: i])
        labels.append(returns[i])
    return np.array(features), np.array(labels)

# Ensures each window has mean 0, standard deviation 1
# Should help KNN recognise patterns more easily
def normalise_patterns(features: np.ndarray) -> np.ndarray:
    # Calculates the mean
    mean = features.mean(axis = 1, keepdims = True)
    # Calculates the standard deviation
    std = features.std(axis = 1, keepdims = True) + 1e-8
    # Returns the normalised
    # Z = mu - x / sigma
    return (features - mean) / std


# Scoring Metric (RMSE)
def rmse(label, prediction):
    return np.sqrt(np.mean((label - prediction) ** 2))

#Splits the data (features and labels) into training, validation and testing
def data_split(features, labels, split_ratio = (0.70, 0.15, 0.15)):
    # Works for numpy arrays and panda dataframes
    n = len(features)
    # Find the end of the training data, assumed to be 1 before start of validation
    train_end = int(n * split_ratio[0])
    # Find end of validation, assumed to be 1 before start of testing - testing
    # ratio value isn't actually used, inputted values must sum to 1 for this to work
    val_end = train_end + int(n * split_ratio[1])

    # Support both numpy arrays and DataFrames
    def _slice(arr, start, end):
        if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            return arr.iloc[start:end]
        return arr[start:end]

    f_train = _slice(features, 0, train_end)
    l_train = _slice(labels, 0, train_end)
    f_val = _slice(features, train_end, val_end)
    l_val = _slice(labels, train_end, val_end)
    f_test = _slice(features, val_end, n)
    l_test = _slice(labels, val_end, n)

    return f_train, l_train, f_val, l_val, f_test, l_test

# Expanding window walk forward validation - repeatedly splits dataframe into training and test window
# On each iteration, trains on past data, standardises uses training data, predicts on test window and 
# stores the predictions as well as the actual data
# Inputs: features and labels are both numpy arrays
#   prediction algorithm is the prediction logic from the model files
#   data split ratios are for training validation and testing
#   k is often not used, for knn it's referring to the size of each neighbourhood
# Outputs: numpy arrays of the predicted values, the actual values and the rmse errors at each prediction
def walk_forward_validation(features, labels, prediction_algorithm,
                             data_split_ratios = (0.7, 0.15, 0.15), k = 5):
    # Standardizes within the window, only on that training data to avoid any leakage
    def _standardise(train, test):
        mean = train.mean(axis = 0)
        std = train.std(axis = 0) + 1e-8
        return (train - mean) / std, (test - mean) / std

    # Ensure numpy - any panda dataframes get converted to numpy arrays
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    # Split the data into training and testing
    n = len(features)
    training_window = int(n * data_split_ratios[0])
    testing_window = max(int(n * data_split_ratios[2]), 1)

    # Prepare the prediction and actual data lists
    predictions, actuals = [], []

    # For each training window, calculate predictions, starting from predicting 1 day ahead
    # incrementing the n-ahead predictions one at a time - still never using testing data for 
    # training, only evaluation
    for start in range(0, n - training_window - testing_window + 1, testing_window):
        train_end = start + training_window
        test_end = train_end + testing_window

        # Get the feature and label training and testing data
        f_train = features[start:train_end]
        l_train = labels[start:train_end]
        f_test = features[train_end:test_end]
        l_test = labels[train_end:test_end]

        # Standardize
        f_train_s, f_test_s = _standardise(f_train, f_test)

        # Make predictions
        preds = prediction_algorithm(f_train_s, l_train, f_test_s, k = k)

        # Now save the predictions and actual values
        predictions.extend(preds)
        actuals.extend(l_test)

    # Convert all outputs to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    rmse_vals = np.sqrt(np.mean((predictions - actuals) ** 2, axis = 0))

    return predictions, actuals, rmse_vals


def predict_future_rows(data_before_targets, data_after_targets, features_full, features_labeled,
                        labels_labeled, predict_fn, n_steps, **predict_kwargs):
    """
    If data_before_targets has more rows than data_after_targets, predict on those missing rows.
    Returns (future_preds, future_opens, future_dates) or (None, None, None) if no future rows.
    """
    n_future = len(data_before_targets) - len(data_after_targets)
    if n_future <= 0:
        return None, None, None

    # Standardize using the labeled data statistics
    mean = features_labeled.mean(axis = 0)
    std = features_labeled.std(axis = 0) + 1e-8
    features_labeled_s = (features_labeled - mean) / std
    features_full_s = (features_full    - mean) / std

    # Predict using full feature set (so sequence models have context)
    all_preds = predict_fn(features_labeled_s, labels_labeled,
                              features_full_s, **predict_kwargs)
    future_preds = np.asarray(all_preds)[-n_future:]

    return future_preds, data_before_targets["open"].values[-n_future:], data_before_targets.index[-n_future:]

# Specific WFR for Decision Tree Regression
# Inputs: features and labels just as for the above wfv
#   estimator class can be any regressor with fit/predict methods, but is built for
#    the DecisionTreeRegressor model
#   Data split ratios are training - validation - testing splits
#   k goes unused in this instance, included for ease of use when executing multiple
#    algorithms sequentially for testing / evaluations
#   Estimator kwargs are passed into the estimator models constructor (in this case
#    being passed into the Decision Tree Regressors constructor)
def dtr_walk_forward_validation(features, labels, estimator_class,
                                 data_split_ratios = (0.8, 0.1, 0.1),
                                 k = None, **estimator_kwargs):
    # Standardise function used to explicitly seperate training and testing data, 
    # standardizing them and returning the standardized values
    def _standardise(train, test):
        mean = train.mean(axis = 0)
        std = train.std(axis = 0) + 1e-8
        return (train - mean) / std, (test - mean) / std

    # Ensure the inputs are numpy arrays, if panda dataframes then convert to numpy arrays
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    # Split data according to ratios
    n = len(features)
    training_window = int(n * data_split_ratios[0])
    testing_window = max(int(n * data_split_ratios[2]), 1)

    # Record the predictions, actual data
    # test indices is used to keep track of which rows are used for the training and validation
    # to keep track of which are to be used for testing
    predictions, actuals, test_indices = [], [], []

    for start in range(0, n - training_window - testing_window + 1, testing_window):
        train_end = start + training_window
        test_end = min(train_end + testing_window, n)

        f_train, l_train = features[start:train_end], labels[start:train_end]
        f_test,  l_test = features[train_end:test_end], labels[train_end:test_end]

        f_train_s, f_test_s = _standardise(f_train, f_test)

        # Create the model and get its predictions
        model = estimator_class(**estimator_kwargs)
        model.fit(f_train_s, l_train)
        preds = model.predict(f_test_s)

        # Record the results, including which values are for testing + validation
        predictions.extend(preds)
        actuals.extend(l_test)
        test_indices.extend(range(train_end, test_end))

    # Ensure all outputs are numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    rmse_per_col = np.sqrt(np.mean((predictions - actuals) ** 2, axis = 0))

    return predictions, actuals, rmse_per_col, test_indices

# --- Feature Engineering ---
def feature_engineering(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to meddle with without damaging the original
    df = data_frame.copy()

    # Use the returns from WRDS
    df["return"] = df["DlyPrcRet"]
    # Calculate a log return from WRDS' daily price data
    df["log return"] = np.log(df["DlyPrcInd"] / df["DlyPrcInd"].shift(1))

    # 100+ probably never used
    # Calculates the moving average (ma), volatility and momentum of the market over the window w
    for w in [5, 10, 25, 50, 100, 200, 250]:
        df[f"ma_{w}"] = df["DlyPrcInd"].rolling(w).mean()
        df[f"volatility_{w}"] = df["return"].rolling(w).std()
        df[f"momentum_{w}"] = df["DlyPrcInd"] - df["DlyPrcInd"].shift(w)

    # Calculate lagged returns
    for lag in range(1, 26):
        df[f"return_t-{lag}"] = df["return"].shift(lag)

    # Drop any values of +-infinity or NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# --- Other ---
def reconstruct_prices(start_price, returns):
    prices, current = [], start_price
    for r in returns:
        current = current * (1 + r)
        prices.append(current)
    return np.array(prices)

# Save the old style matplotlib plots with rmse
def save_results(predictions, actuals, rmse = 0, model_name = "model"):
    import matplotlib.pyplot as plt
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize = (20, 10))
    plt.plot(actuals,     label = "Actual closing")
    plt.plot(predictions, label = "Predicted closing")
    plt.legend()
    plt.title(f"{model_name}  RMSE = {rmse:.4f}")
    plt.savefig(f"{model_name}_rmse-{rmse}_{ts}.png")
    plt.close()

# Just uses the pandas to_csv function to save the output, but adds a timestamp to the name
def save_models_data(model_name, df):
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"Results/{model_name}_{time_stamp}.csv")
    df.to_csv(f"Newest Results/{model_name}.csv")

# To load the previously saved data into a dataframe - just inverse of the function above
def load_models_data(file_name):
    df = pd.read_csv()

# --- Matplot Lib Plots - Line Graphs ---
def plot_all_mp_graphs(model_results, model_name = "test", save = False, show = True):    
    # The price cols are the column names for the predicted prices
    # Rel cols are the predicted relative proportions (returns)
    # d_c referring to the daisy chained versions, using true n-step ahead - shows alternative
    #  interpretation to models accuracy and allows models to predict future data
    price_cols = ["open", "high", "low", "close"]
    rel_cols = ["open_rel", "high_rel", "low_rel", "close_rel"]
    d_c_price_cols = ["daisy_chained_open", "daisy_chained_high", "daisy_chained_low", "daisy_chained_close"] 
    d_c_rel_cols = ["daisy_chained_open_rel", "daisy_chained_high_rel", "daisy_chained_low_rel", "daisy_chained_close_rel"]

    # Create a plot of graphs 4 tall, 4 wide with each figure being of size 16x14
    # Rows correspond to the price types - Open High Low Close
    # Columns correspond to: returns, absolute prices (1-step ahead), daisy chained returns, daisy chained absolute prices (true n-step ahead)
    fig, axes = plt.subplots(4, 4, figsize = (16, 14))

    # Iterate through each row of plots
    # Where zip() groups together the 1 and n step ahead returns and prices
    for row, (rel_col, abs_col, d_c_rel_col, d_c_abs_col) in enumerate(zip(rel_cols, price_cols, d_c_rel_cols, d_c_price_cols)):
        # Select the 4 axes in the current row
        ax_rel, ax_abs, ax_dc_rel, ax_dc_abs = axes[row][0], axes[row][1], axes[row][2], axes[row][3]

        for name, df in model_results.items():
            # Ensure the dataframe contains all the required columns before plotting
            required_cols = [rel_col, abs_col, d_c_rel_col, d_c_abs_col]
            if all(col in df.columns for col in required_cols):
                # Plot the returns
                ax_rel.plot(df.index, df[rel_col].values, label = name)
                # Plot absolute predicted prices
                ax_abs.plot(df.index, df[abs_col].values, label = name)
                # Plot daisy chained returns (should be equal to above returns, used for sanity checks)
                ax_dc_rel.plot(df.index, df[d_c_rel_col].values, label = name)
                # Plot daisy chained absolute prices
                ax_dc_abs.plot(df.index, df[d_c_abs_col].values, label = name)
        
        # Now set the titles and all the labels for each subplot in this row

        # Returns plot
        ax_rel.set_title(f"Predicted {rel_col.capitalize()} (relative to open)")
        ax_rel.set_xlabel("Date")
        ax_rel.legend(fontsize = 6)

        # Absolute prices plot
        ax_abs.set_title(f"Predicted {abs_col.capitalize()} (price)")
        ax_abs.set_xlabel("Date")
        ax_abs.legend(fontsize = 6)

        # Daisy-Chained returns plot
        ax_dc_rel.set_title(f"Predicted {d_c_rel_col.capitalize()} (relative to open)")
        ax_dc_rel.set_xlabel("Date")
        ax_dc_rel.legend(fontsize = 6)
        
        # Daisy-Chained prices plot
        ax_dc_abs.set_title(f"Predicted {d_c_abs_col.capitalize()} (price)")
        ax_dc_abs.set_xlabel("Date")
        ax_dc_abs.legend(fontsize = 6)

    # Adjust the layout to prevent overlapping titles / labels
    plt.tight_layout()

    if save:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save as the plot as an image
        fig.savefig(f"Results/{model_name}_{time_stamp}.png")
        fig.savefig(f"Newest Results/{model_name}.png")

    if show:
        # Display the figure
        plt.show()

def plot_mp_graphs(model_results, model_name = "test", save = False, show = True):    
    # The price cols are the column names for the predicted prices
    # Rel cols are the predicted relative proportions (returns)
    # d_c referring to the daisy chained versions, using true n-step ahead - shows alternative
    #  interpretation to models accuracy and allows models to predict future data
    price_cols = ["open", "high", "low", "close"]
    rel_cols = ["open_rel", "high_rel", "low_rel", "close_rel"]
    d_c_price_cols = ["daisy_chained_open", "daisy_chained_high", "daisy_chained_low", "daisy_chained_close"] 
    d_c_rel_cols = ["daisy_chained_open_rel", "daisy_chained_high_rel", "daisy_chained_low_rel", "daisy_chained_close_rel"]

    # Create a plot of graphs 4 tall, 4 wide with each figure being of size 16x14
    # Rows correspond to the price types - Open High Low Close
    # Columns correspond to: returns, absolute prices (1-step ahead), daisy chained returns, daisy chained absolute prices (true n-step ahead)
    fig, axes = plt.subplots(4, 4, figsize = (16, 14))

    # Iterate through each row of plots
    # Where zip() groups together the 1 and n step ahead returns and prices
    for row, (rel_col, abs_col, d_c_rel_col, d_c_abs_col) in enumerate(zip(rel_cols, price_cols, d_c_rel_cols, d_c_price_cols)):
        # Select the 4 axes in the current row
        ax_rel, ax_abs, ax_dc_rel, ax_dc_abs = axes[row][0], axes[row][1], axes[row][2], axes[row][3]
        name = model_name
        df = model_results
    
        # Ensure the dataframe contains all the required columns before plotting
        required_cols = [rel_col, abs_col, d_c_rel_col, d_c_abs_col]
        if all(col in df.columns for col in required_cols):
            # Plot the returns
            ax_rel.plot(df.index, df[rel_col].values, label = name)
            # Plot absolute predicted prices
            ax_abs.plot(df.index, df[abs_col].values, label = name)
            # Plot daisy chained returns (should be equal to above returns, used for sanity checks)
            ax_dc_rel.plot(df.index, df[d_c_rel_col].values, label = name)
            # Plot daisy chained absolute prices
            ax_dc_abs.plot(df.index, df[d_c_abs_col].values, label = name)
        
        # Now set the titles and all the labels for each subplot in this row

        # Returns plot
        ax_rel.set_title(f"Predicted {rel_col.capitalize()} (relative to open)")
        ax_rel.set_xlabel("Date")
        ax_rel.legend(fontsize = 6)

        # Absolute prices plot
        ax_abs.set_title(f"Predicted {abs_col.capitalize()} (price)")
        ax_abs.set_xlabel("Date")
        ax_abs.legend(fontsize = 6)

        # Daisy-Chained returns plot
        ax_dc_rel.set_title(f"Predicted {d_c_rel_col.capitalize()} (relative to open)")
        ax_dc_rel.set_xlabel("Date")
        ax_dc_rel.legend(fontsize = 6)
        
        # Daisy-Chained prices plot
        ax_dc_abs.set_title(f"Predicted {d_c_abs_col.capitalize()} (price)")
        ax_dc_abs.set_xlabel("Date")
        ax_dc_abs.legend(fontsize = 6)

    # Adjust the layout to prevent overlapping titles / labels
    plt.tight_layout()

    if save:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save as the plot as an image
        fig.savefig(f"Results/{model_name}_{time_stamp}.png")

    if show:
        # Display the figure
        plt.show()

# --- Pyplot Plots - Candlesticks ---
def plot_single_candlestick_graph(df, model_name = "test", save = False, show = True):
    # Create a candlestick graph using the dataframe inputted
    fig = go.Figure(data = [go.Candlestick(
                    x = df.index,
                    open = df["open"],
                    high = df["high"],
                    low = df["low"],
                    close = df["close"])])

    if save:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Saves as an html file to maintain interactability - zooming, hover to see labels etc
        fig.write_html(f"Results/{model_name}_{time_stamp}.html")
        # Save as an image as well for uniformity
        fig.write_image(f"Results/{model_name}_{time_stamp}.png")

    if show:
        # Display the figure
        fig.show()

def plot_candlestick_graphs(df, model_name = "test", save = False, show = True):
    # Ensure required columns exist in the inputted df before plotting
    required_cols = [
        "open", "high", "low", "close",
        "daisy_chained_open", "daisy_chained_high",
        "daisy_chained_low", "daisy_chained_close"
    ]
    
    # Exits the function if the required data isn't in the inputted df
    if not all(col in df.columns for col in required_cols):
        print("Missing required columns for candlestick plot")
        return

    # Create a subplot layout of 1 row, 2 columns with appropriate titles - regular and daisy chained prices
    fig = make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = ("Regular Prices", "Daisy-Chained Prices")
    )

    # Left plot - the regular OHLC (Open High Low Close) prices
    fig.add_trace(
        go.Candlestick(
            x = df.index,
            open = df["open"],
            high = df["high"],
            low = df["low"],
            close = df["close"],
            name = "Regular"
        ),
        row = 1, col = 1
    )

    # Right plot - the daisy chained OHLC (true n-step ahead)
    fig.add_trace(
        go.Candlestick(
            x = df.index,
            open = df["daisy_chained_open"],
            high = df["daisy_chained_high"],
            low = df["daisy_chained_low"],
            close = df["daisy_chained_close"],
            name = "Daisy Chained"
        ),
        row = 1, col = 2
    )


    # Now add names to the axes
    fig.update_layout(
        title = "Candlestick Comparison",
        xaxis_title = "Date",
        yaxis_title = "Price",
        xaxis2_title = "Date",
        yaxis2_title = "Price"
    )
    if save:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Saves as an html file to maintain interactability - zooming, hover to see labels etc
        fig.write_html(f"Results/{model_name}_{time_stamp}.html")
        # Save as an image as well for uniformity - required kaleido to be installed so add error handling
        try:
            fig.write_image(f"Results/{model_name}_{time_stamp}.png")
        except:
            print(f"The model {model_name} could not be saved to a png file.")

    if show:
        # Show the figure
        fig.show()
