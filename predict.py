import yfinance as yf
import pandas as pd
import joblib
from prepare_data import add_features

# Load the trained model
model = joblib.load("stock_model.pkl")

def predict_stock(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    if df.empty or len(df) < 30:
        return {"Ticker": ticker, "Error": "Insufficient data"}

    df = add_features(df)
    latest = df.iloc[[-1]]
    X = latest[['SMA_20', 'EMA_20', 'RSI']]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "Ticker": ticker,
        "Prediction": "Will Rise 5–10%" if prediction == 1 else "Not Likely",
        "Confidence": round(probability * 100, 2)
    }

if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g., AAPL): ").upper()
    result = predict_stock(ticker)
    print(result)