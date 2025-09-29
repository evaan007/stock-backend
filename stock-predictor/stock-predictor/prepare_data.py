import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

tickers = ['AAPL', 'MSFT', 'TSLA']
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

print("Script started...")  # CHECKPOINT 1

def label_data(df):
    df['Future_Close'] = df['Close'].shift(-14)
    df['Price_Change'] = (df['Future_Close'] - df['Close']) / df['Close']
    df['Signal'] = df['Price_Change'].apply(
        lambda x: 'Yes' if x >= 0.05 else ('No' if x <= -0.02 else 'Hold')
    )
    return df

def add_indicators(df):
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

for ticker in tickers:
    print(f"Downloading: {ticker}")
    df = yf.download(ticker, start=start_date, end=end_date)
    print(f"{ticker} - Rows fetched: {len(df)}")
    if df.empty:
        print(f"No data for {ticker}")
        continue
    df = add_indicators(df)
    df = label_data(df)
    df.dropna(inplace=True)
    filename = f"{ticker}_labeled.csv"
    df.to_csv(filename)
    print(f"Saved {filename}")

print("Script completed!")  # CHECKPOINT 2

