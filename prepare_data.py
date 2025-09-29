import yfinance as yf
import pandas as pd

# Download historical stock data
def download_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Add technical indicators
def add_features(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df.dropna(inplace=True)
    return df

# Add label: 1 if price goes up 5%+ in next 10 days
def add_labels(df):
    future_price = df['Close'].shift(-10)
    pct_change = (future_price - df['Close']) / df['Close']
    df['Label'] = (pct_change >= 0.05).astype(int)
    df.dropna(inplace=True)
    return df

# Full pipeline
def prepare_data(ticker):
    df = download_data(ticker)
    df = add_features(df)
    df = add_labels(df)
    return df

# Example usage
if __name__ == "__main__":
    ticker = "AAPL"
    df = prepare_data(ticker)
    df.to_csv("prepared_data.csv", index=False)
    print("Data saved to prepared_data.csv")