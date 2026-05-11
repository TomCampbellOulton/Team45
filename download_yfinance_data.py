import yfinance as yf
import pandas as pd

# Download data
ticker = "^GSPC"
df = yf.download(ticker, start = "2000-01-01", auto_adjust = False)

# Reset index
df = df.reset_index()

# Rename columns
df.rename(columns = {
    "Date": "DlyCalDt",
    "Close": "Close",
    "Adj Close": "AdjClose",
    "Volume": "DlyUsdCnt"
}, inplace = True)

# Returns
df["DlyTotRet"] = df["AdjClose"].pct_change()
df["DlyPrcRet"] = df["Close"].pct_change()
df["DlyIncRet"] = df["DlyTotRet"] - df["DlyPrcRet"]

# Index levels
df["DlyTotInd"] = (1 + df["DlyTotRet"]).cumprod()
df["DlyPrcInd"] = (1 + df["DlyPrcRet"]).cumprod()
df["DlyIncInd"] = (1 + df["DlyIncRet"]).cumprod()

# Formatting + static fields
df["INDNO"] = 1
df["YYYYMMDD"] = df["DlyCalDt"].dt.strftime("%Y%m%d")

df["DlyUsdVal"] = df["Close"] * df["DlyUsdCnt"]

df["DlyTotCnt"] = ""
df["DlyTotVal"] = ""
df["DlyEligCnt"] = 500
df["DlyWgtAmt"] = ""

df["INDFAM"] = "SP"
df["IndFamType"] = "Equity"
df["IndNm"] = "S&P 500"
df["IndBegDt"] = "19570304"
df["IndEndDt"] = ""

df["BaseLvl"] = 10
df["BaseDt"] = "19570304"
df["FreqAvail"] = "D"
df["WeightType"] = "MarketCap"
df["CntValType"] = "USD"
df["PortNum"] = 1

# Column order
cols = [
    "INDNO","YYYYMMDD","DlyCalDt",
    "DlyTotRet","DlyTotInd",
    "DlyPrcRet","DlyPrcInd",
    "DlyIncRet","DlyIncInd",
    "DlyUsdCnt","DlyUsdVal",
    "DlyTotCnt","DlyTotVal",
    "DlyEligCnt","DlyWgtAmt",
    "INDFAM","IndFamType","IndNm",
    "IndBegDt","IndEndDt",
    "BaseLvl","BaseDt",
    "FreqAvail","WeightType",
    "CntValType","PortNum"
]

df = df[cols]

# Save file
df.to_csv("Data/S&P 500 Composite.csv", index = False)

print("File created: sp500_filled.csv")