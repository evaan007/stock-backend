from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# RSI calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Main stock analysis function
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", interval="1d")

        if hist.empty or len(hist) < 30:
            return {"Ticker": ticker, "Error": "Insufficient or no data", "Final Signal": "Error"}

        hist['EMA20'] = hist['Close'].ewm(span=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = compute_rsi(hist['Close'])

        resistance = hist['Close'][:-5].max()
        breakout = hist['Close'].iloc[-1] > resistance
        retest = hist['Close'].iloc[-5:-1].min() >= resistance * 0.97
        bullish = hist['Close'].iloc[-1] > hist['Open'].iloc[-1]
        ema_cross = hist['EMA20'].iloc[-1] > hist['SMA50'].iloc[-1]
        rsi_val = round(hist['RSI'].iloc[-1], 2)
        rsi_ok = 50 < rsi_val < 70
        volume_spike = hist['Volume'].iloc[-1] > 1.5 * hist['Volume'].rolling(10).mean().iloc[-1]

        passed = sum([breakout, retest, bullish, ema_cross, rsi_ok, volume_spike])
        signal = "Buy" if passed >= 4 else "Hold" if passed >= 2 else "Sell"

        return {
            "Ticker": ticker,
            "EMA20": round(hist['EMA20'].iloc[-1], 2),
            "SMA50": round(hist['SMA50'].iloc[-1], 2),
            "RSI": rsi_val,
            "Breakout": breakout,
            "Retest": retest,
            "Bullish Pattern": bullish,
            "EMA Crossover": ema_cross,
            "Volume Spike": volume_spike,
            "Final Signal": signal
        }

    except Exception as e:
        return {"Ticker": ticker, "Error": str(e), "Final Signal": "Error"}

# Flask route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        tickers_input = request.form.get("tickers", "")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        result = [analyze_stock(ticker) for ticker in tickers]
    return render_template("index.html", result=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)