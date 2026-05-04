#https://www.youtube.com/watch?v=AXBhrLongC8

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

start_date = '2010-01-01'
end_date = '2025-12-31'

ret_cols = ['DlyPrcRet']
columns = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
price_cols = ['open', 'high', 'low', 'close']
colors = {'open': 'blue', 'close': 'red', 'high': 'green', 'low': 'orange'}


def get_data_yf(start_date, end_date):
    df = yf.download('^GSPC', start_date, end_date)
    df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    return df

def lagit(df, lags): 
    feature_cols = []
    for col in columns:
        for col in range(1, lags+1):
            name = f'ret_Lag_{col}'
            df[name] = df['DlyPrcRet'].shift(col) #one step ahead
            feature_cols.append(name)
    return feature_cols

def to_weekly(daily_series):
    return daily_series.resample('W').last()

def svr(df, train, test, lagnames):
    plt.figure(figsize=(12, 6))

    data = {}

    for price_col in price_cols:
        train = train.copy()
        test = test.copy()
        model = SVR(kernel='rbf', C=10, gamma='scale' , epsilon=0.1)

        scaler= StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler.fit_transform(train[lagnames])
        X_test = scaler.transform(test[lagnames])
        y_train = scaler_y.fit_transform(train[['DlyPrcRet']]).ravel()
        model.fit(X_train, y_train)

        test['prediction_SVR'] = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
        train['train_SVR'] = scaler_y.inverse_transform(model.predict(X_train).reshape(-1, 1)).ravel() 

        predicted_returns = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()

        last_known_price = train[price_col].iloc[-1]

        reconstructed_values = [last_known_price]
        for ret in predicted_returns:
            reconstructed_values.append(reconstructed_values[-1] * (1 + ret)) #take prev price * by 1 +ret
        reconstructed = pd.Series(reconstructed_values[1:], index=test.index) #skip first element since we only want predicted prices, test dates as index
        
        rmse = np.sqrt(mean_squared_error(test[price_col], reconstructed))
        r_squared = r2_score(test[price_col], reconstructed)
        mae = mean_absolute_error(test[price_col], reconstructed)
        dir_acc = np.mean(np.sign(test[price_col].diff()) == np.sign(reconstructed.diff()))

        print(price_col)
        print("Root Mean Square Error (RMSE):", rmse)
        print("R^2 Score:", r_squared)
        print("Mean Absolute Error (MAE):", mae)
        print("Directional Accuracy:", dir_acc, "\n")

        reconstructed_weekly = to_weekly(reconstructed)
        data[price_col] = reconstructed_weekly

        plt.plot(reconstructed_weekly.index,reconstructed_weekly,color=colors[price_col], label=f'{price_col} (weekly)')
        plt.plot(test.index, test[price_col], color=colors[price_col], linestyle='--', label=f'{price_col} actual (daily)')

    data = pd.DataFrame(data)
    data.to_csv(f'weekly_predictions.csv')


    plt.title('SVR Predictions - OHLC')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Gets the data from the csv file from WRDS and combines with yfinance data
# using yfinance's ratios for OHLC to multiply WRDS' daily value by (WRDS is more reliable)
def get_data(start_date, end_date, data_file="S&P 500 Composite.csv"):
    # Reads only relevant columns from WRDS data - date, daily return and daily price
    df = pd.read_csv(data_file, usecols=["YYYYMMDD", "DlyPrcRet", "DlyPrcInd"])

    # Convert the string date to pandas datetime
    df["Date"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
    # Use datetime as the index for dataframe
    df.set_index("Date", inplace=True)

    # Get the start and end date in pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)

    # Filter the data to only be between the start and end date inclusively
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    # Sort the data by date
    filtered_df.sort_index(inplace=True)

    # Fetch OHLC ratios from yfinance
    y_finance = get_data_yf(start_date, end_date)
    y_finance = y_finance.reindex(filtered_df.index)

    # Relative effectively being return, NOTE - open not used as WRDS
    # is being used as the 'open' prices - yfinance's raw prices not used, only
    # their daily ratios relative to market open
    relative_high  = y_finance["high"]  / y_finance["open"]
    relative_low   = y_finance["low"]   / y_finance["open"]
    relative_close = y_finance["close"] / y_finance["open"]

    # Use WRDS data as the open price for the day
    base_price = filtered_df["DlyPrcInd"]
    # Then use YFinance to fill in the high, low, close - not using yfinances exact values
    # but rather using their ratios for high, low, close relative to open and then applying
    # that ratio to the open (WRDS data)
    filtered_df["open"]  = base_price
    filtered_df["high"]  = base_price * relative_high
    filtered_df["low"]   = base_price * relative_low
    filtered_df["close"] = base_price * relative_close

    # Drop non-numeric columns (YYYYMMDD, etc.)
    filtered_df = filtered_df.drop(columns=["YYYYMMDD"], errors="ignore")
    filtered_df = filtered_df.select_dtypes(include=["number"])

    return filtered_df


def main():
    df = get_data(start_date, end_date)
    lagnames = lagit(df, 5)
    df.dropna(inplace=True) #drop NaN values
    train,test = train_test_split(df, test_size=0.15, shuffle=False, random_state=0)
    svr(df, train, test, lagnames)

if __name__ == "__main__":
    main()
    

