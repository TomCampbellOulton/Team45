import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt


#edit dates here
start_date='2026-05-01'
end_date='2027-05-01'

ret_cols = ['DlyPrcRet']
columns = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
price_cols = ['open', 'high', 'low', 'close']
colors = {'open': 'blue', 'close': 'red', 'high': 'green', 'low': 'orange'}


def get_data_yf(start_date, end_date):
    # progress=False removes the download bar text/warnings
    df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    df.columns = df.columns.get_level_values(0)
        
    df.columns = df.columns.str.lower()
    return df

def lagit(df, lags):
    """
    Creates lagged returns as features for the model.
    """
    feature_cols = []
    for i in range(1, lags + 1):
        name = f'ret_Lag_{i}'
        df[name] = df['DlyPrcRet'].shift(i)  # one step ahead
        feature_cols.append(name)
    return feature_cols

def to_weekly(daily_series):
    return daily_series.resample('W-SUN').last()

def predict_future_range(df, train, test, model, scaler, scaler_y, lagnames, price_col, start_date, end_date, freq='D'):
    
    # Predict from start_date to end_date
    # freq: 'D' for daily, 'W' for weekly
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Get the last row of test data
    last_row = test.iloc[-1].copy()
    last_price = last_row[price_col]
    
    if freq == 'D':
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    elif freq == 'W':
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    
    predictions = []
    
    #step into t+1 by taking the last known lag values and rolling in the last actual daily return
    current_lags = last_row[lagnames].values.reshape(1, -1)
    current_lags = np.roll(current_lags, 1)
    current_lags[0, 0] = last_row['DlyPrcRet']
    
    for i in range(len(date_range)):
        X_future = scaler.transform(current_lags)
        pred_return_scaled = model.predict(X_future)
        pred_return = scaler_y.inverse_transform(pred_return_scaled.reshape(-1, 1))[0][0]
        
        # Calculate next price
        next_price = last_price * (1 + pred_return)
        predictions.append(next_price)
        
        # Update for next iteration
        last_price = next_price
        current_lags = np.roll(current_lags, 1)
        current_lags[0, 0] = pred_return
    
    return pd.Series(predictions, index=date_range)


def svr(df, train, test, lagnames):
    data = {}
    plt.figure(figsize=(14, 6))

    for price_col in price_cols:
        train_c = train.copy()
        test_c = test.copy()
        model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)

        scaler = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler.fit_transform(train_c[lagnames].values)
        X_test = scaler.transform(test_c[lagnames].values)
        y_train = scaler_y.fit_transform(train_c[['DlyPrcRet']].values).ravel()
        model.fit(X_train, y_train)

        test_c['prediction_SVR'] = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()

        #predict from May 1 2026 to May 1 2027
        future_predictions = predict_future_range(df, train_c, test_c, model, scaler, scaler_y, lagnames, price_col, start_date=start_date, end_date=end_date, freq='W')
        
        data[price_col] = future_predictions
        
        plt.plot(future_predictions.index, future_predictions.values, color=colors[price_col], label=f'{price_col} (weekly)')
    
    plt.title('SVR Predictions - OHLC (Weekly)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    predictions_df = pd.DataFrame(data)
    predictions_df.to_csv('year_predictions.csv')


def get_data(start_date, end_date, data_file="S&P 500 Composite.csv"):
    # Reads only relevant columns from WRDS data - date, daily return and daily price
    df = pd.read_csv(data_file, usecols=["YYYYMMDD", "DlyPrcRet", "DlyPrcInd"])

    # Convert the string date to pandas datetime
    df["Date"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
    df.set_index("Date", inplace=True)

    # Get the start and end date in pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)

    # Filter the data to only be between the start and end date inclusively
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    filtered_df.sort_index(inplace=True)

    # Fetch OHLC ratios from yfinance
    y_finance = get_data_yf(start_date, end_date)
    y_finance = y_finance.reindex(filtered_df.index)

    relative_high  = y_finance["high"]  / y_finance["open"]
    relative_low   = y_finance["low"]   / y_finance["open"]
    relative_close = y_finance["close"] / y_finance["open"]

    base_price = filtered_df["DlyPrcInd"]
    
    filtered_df["open"]  = base_price
    filtered_df["high"]  = base_price * relative_high
    filtered_df["low"]   = base_price * relative_low
    filtered_df["close"] = base_price * relative_close

    # Drop non-numeric columns
    filtered_df = filtered_df.drop(columns=["YYYYMMDD"], errors="ignore")
    filtered_df = filtered_df.select_dtypes(include=["number"])

    return filtered_df


def main():
    df = get_data('2010-01-01', '2025-12-31')
    lagnames = lagit(df, 5)
    
    df.dropna(inplace=True) # drop NaN values after lagging
    
    train, test = train_test_split(df, test_size=0.15, shuffle=False, random_state=0)
    svr(df, train, test, lagnames)

if __name__ == "__main__":
    main()