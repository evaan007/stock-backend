import yfinance as yf
import pandas as pd

# Step 1: Download historical stock data
def download_data(ticker):
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

# Step 2: Add technical indicators
def add_features(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df.dropna(inplace=True)
    return df

# Step 3: Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Step 4: Label data: 1 if stock rises 5%+ in 10 days
def add_labels(df):
    future_price = df['Close'].shift(-10)
    pct_change = (future_price - df['Close']) / df['Close']
    df['Label'] = (pct_change >= 0.05).astype(int)
    df.dropna(inplace=True)
    return df

# Step 5: Run pipeline
if __name__ == "__main__":
    ticker = "AAPL"  # You can change this or use a list
    df = download_data(ticker)
    df = add_features(df)
    df = add_labels(df)
    df.to_csv("prepared_data.csv", index=False)
    print("Data saved to prepared_data.csv")