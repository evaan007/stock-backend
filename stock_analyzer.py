import yfinance as yf
import pandas as pd

def analyze_multiple_stocks(ticker_list):
    results = []

    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo", interval="1d")

            if hist.empty:
                results.append({ticker: "No data found."})
                continue

            hist['EMA20'] = hist['Close'].ewm(span=20).mean()
            hist['EMA50'] = hist['Close'].ewm(span=50).mean()
            hist['RSI'] = compute_rsi(hist['Close'])

            resistance = hist['Close'].rolling(window=20).max().iloc[-10]
            last_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(window=10).mean().iloc[-1]

            breakout = last_close > resistance
            retest = prev_close < resistance and last_close > resistance
            rsi_ok = 50 <= hist['RSI'].iloc[-1] <= 70
            ema_crossover = hist['EMA20'].iloc[-1] > hist['EMA50'].iloc[-1]
            volume_spike = volume > avg_volume * 1.3
            bullish_candle = hist['Close'].iloc[-1] > hist['Open'].iloc[-1]

            conditions = {
                "Breakout": breakout,
                "Retest": retest,
                "Bullish Candlestick": bullish_candle,
                "RSI (50–70)": rsi_ok,
                "EMA Crossover": ema_crossover,
                "Volume Spike": volume_spike,
            }

            passed = [k for k, v in conditions.items() if v]
            if len(passed) >= 4:
                result = f"BUY SIGNAL: {ticker} meets {len(passed)}/6 conditions"
            else:
                result = f"HOLD: {ticker} meets only {len(passed)}/6 conditions"

            results.append({
                "Ticker": ticker,
                "Result": result,
                "Details": conditions
            })

        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Result": "Error",
                "Details": str(e)
            })

    return results

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))