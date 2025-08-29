import streamlit as st
import yfinance as yf
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
import time
import numpy as np

# Try importing plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Technical Analyzer Class
class TechnicalAnalyzer:
    def __init__(self, window_sr=20, rsi_window=14, atr_window=14, bb_window=20, bb_std=2):
        self.window_sr = window_sr
        self.rsi_window = rsi_window
        self.atr_window = atr_window
        self.bb_window = bb_window
        self.bb_std = bb_std

    def support_resistance(self, df):
        support = df['Low'].rolling(self.window_sr).min().iloc[-1].item()
        resistance = df['High'].rolling(self.window_sr).max().iloc[-1].item()
        return support, resistance

    def rsi(self, df):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(self.rsi_window).mean()
        loss = -delta.clip(upper=0).rolling(self.rsi_window).mean()
        rs = gain / loss
        rsi_val = (100 - (100 / (1 + rs))).iloc[-1].item()
        rsi_slope = ((100 - (100 / (1 + rs))).iloc[-1] - (100 - (100 / (1 + rs))).iloc[-2]).item()
        return rsi_val, rsi_slope

    def macd(self, df, short=12, long=26, signal=9):
        exp1 = df['Close'].ewm(span=short, adjust=False).mean()
        exp2 = df['Close'].ewm(span=long, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_val = macd_line.iloc[-1].item()
        signal_val = signal_line.iloc[-1].item()
        macd_hist = macd_line - signal_line
        hist_slope = (macd_hist.iloc[-1] - macd_hist.iloc[-2]).item()
        return macd_val, signal_val, hist_slope

    def atr(self, df):
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean().iloc[-1].item()

    def bollinger_bands(self, df):
        sma = df['Close'].rolling(self.bb_window).mean()
        std = df['Close'].rolling(self.bb_window).std()
        upper = sma + std * self.bb_std
        lower = sma - std * self.bb_std
        return upper.iloc[-1].item(), lower.iloc[-1].item()

    def trend_filter(self, df):
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1].item()
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1].item()
        return ema50 > ema200, ema50 < ema200

    def generate_signal(self, df, current_price):
        support, resistance = self.support_resistance(df)
        rsi_val, rsi_slope = self.rsi(df)
        macd_val, signal_val, hist_slope = self.macd(df)
        atr_val = self.atr(df)
        bb_upper, bb_lower = self.bollinger_bands(df)
        trend_up, trend_down = self.trend_filter(df)

        if atr_val < (df['Close'].iloc[-1].item() * 0.0015):
            current_price = df['Close'].iloc[-1].item()
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(self.rsi_window).mean()
            loss = -delta.clip(upper=0).rolling(self.rsi_window).mean()
            rsi_val = (100 - (100 / (1 + gain / loss))).iloc[-1].item()
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "support": round(support, 2),
                "resistance": round(resistance, 2),
                "rsi": round(rsi_val, 2),
                "macd": 0.0,
                "signal_line": 0.0,
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "current_price": round(current_price, 2),
                "hold_minutes": 0
            }

        score_buy = 0
        score_sell = 0

        if current_price <= support * 1.005:
            score_buy += 0.35
        if current_price >= resistance * 0.995:
            score_sell += 0.35

        if rsi_val < 35 and rsi_slope > 0.1:
            score_buy += 0.25
        elif rsi_val > 65 and rsi_slope < -0.1:
            score_sell += 0.25

        if macd_val > signal_val and hist_slope > 0.01:
            score_buy += 0.25
        elif macd_val < signal_val and hist_slope < -0.01:
            score_sell += 0.25

        if current_price < bb_lower * 1.005:
            score_buy += 0.2
        if current_price > bb_upper * 0.995:
            score_sell += 0.2

        if score_buy > 0 and not trend_up:
            score_buy *= 0.6
        if score_sell > 0 and not trend_down:
            score_sell *= 0.6

        if score_buy > score_sell + 0.1:
            signal = "BUY"
            confidence = score_buy
            hold_minutes = 20
        elif score_sell > score_buy + 0.1:
            signal = "SELL"
            confidence = score_sell
            hold_minutes = 20
        else:
            signal = "HOLD"
            confidence = max(score_buy, score_sell)
            hold_minutes = 0

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "rsi": round(rsi_val, 2),
            "macd": round(macd_val, 2),
            "signal_line": round(signal_val, 2),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "current_price": round(current_price, 2),
            "hold_minutes": hold_minutes
        }

# Load Gemini API key
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

if not GENAI_API_KEY:
    st.error("Gemini API key not found. Please set it in the .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Streamlit page config
st.set_page_config(page_title="📊 AI Candle Analysis", layout="wide")
st.title("📉 AI-Powered Candlestick Analysis")
st.markdown("Enter a stock symbol to get **entry, stop loss, and target price suggestions** powered by Gemini AI.")

# Cache yfinance
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval)

# Sidebar controls
with st.sidebar:
    st.header("🔧 Settings")
    symbol = st.text_input("Stock symbol (e.g. AAPL, TSLA):", value="AAPL")
    interval = st.selectbox("Candle interval",
        options=["1m","2m","5m","15m","30m","60m","90m","1d","1wk","1mo"],
        index=6
    )
    duration = st.selectbox("Lookback duration",
        options=["1d","5d","1mo","3mo","6mo","1y","2y","5y","ytd","max"],
        index=1
    )
    include_news = st.checkbox("Include news", value=True)
    show_sma = st.checkbox("Show 20-SMA", value=True)
    show_volume = st.checkbox("Show Volume", value=True)
    analyze_button = st.button("🔍 Analyze")
    clear_button = st.button("🧹 Clear")

if clear_button:
    st.session_state.symbol = ""
    st.experimental_rerun()

if not analyze_button:
    st.info("Enter a stock symbol and click **Analyze**.")
    st.image("https://via.placeholder.com/800x400.png?text=Candlestick+Chart+Preview")
else:
    if not symbol:
        st.warning("Please enter a valid stock symbol.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Fetching stock data...")
            data = fetch_stock_data(symbol.upper(), duration, interval)
            progress_bar.progress(25)

            if data.empty:
                status_text.error("No data found for this symbol/interval.")
                progress_bar.empty()
            else:
                # Round OHLC
                for col in ["Open", "High", "Low", "Close"]:
                    data[col] = data[col].round(2)

                if show_sma and len(data) >= 20:
                    data["SMA20"] = data["Close"].rolling(window=20).mean().round(2)

                # Technical Analysis
                analyzer = TechnicalAnalyzer()
                current_price = data["Close"].iloc[-1]
                signals = analyzer.generate_signal(data, current_price)

                progress_bar.progress(50)
                status_text.info("Preparing chart...")

                # Chart
                st.subheader(f"📈 {symbol.upper()} Candlestick Chart ({interval}, {duration})")
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data["Open"],
                        high=data["High"],
                        low=data["Low"],
                        close=data["Close"],
                        name=symbol.upper()
                    ))
                    if show_volume:
                        fig.add_trace(go.Bar(
                            x=data.index, y=data["Volume"],
                            yaxis="y2", name="Volume", opacity=0.3,
                            marker_color="gray"
                        ))
                    if show_sma and "SMA20" in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data["SMA20"],
                            name="20-SMA", line=dict(color="orange", width=2)
                        ))
                    fig.update_layout(
                        xaxis_rangeslider_visible=True,
                        yaxis_title="Price",
                        yaxis2=dict(title="Volume", overlaying="y", side="right") if show_volume else None,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly not installed. Showing raw data.")
                    st.dataframe(data.tail(10))

                # Prepare news
                news_summary = ""
                if include_news:
                    try:
                        ticker = yf.Ticker(symbol.upper())
                        news_items = ticker.news
                        if news_items:
                            news_summary = "\nRecent News:\n"
                            for item in news_items[:3]:
                                pub_date = item.get("providerPublishTime")
                                pub_date = datetime.fromtimestamp(pub_date).strftime("%Y-%m-%d") if pub_date else ""
                                news_summary += f"- {pub_date}: {item.get('title','')} ({item.get('publisher','')})\n"
                    except Exception as e:
                        news_summary = f"\nError fetching news: {e}\n"

                progress_bar.progress(70)
                status_text.info("Analyzing with Gemini...")

                # Simplified Gemini Prompt
                prompt = f"""
You are a financial trading assistant.

Using the summarized technical analysis signals, provide ONLY:
1. Suggested trade entry (buy/sell).
2. Suggested stop loss (SL).
3. Suggested target price (TP).
4. A short reasoning (max 3 sentences).

Technical Analysis Summary:
- Signal: {signals['signal']}
- Confidence: {signals['confidence']}
- Support: {signals['support']}
- Resistance: {signals['resistance']}
- RSI: {signals['rsi']}
- MACD: {signals['macd']}
- Signal Line: {signals['signal_line']}
- Bollinger Upper: {signals['bb_upper']}
- Bollinger Lower: {signals['bb_lower']}
- Current Price: {signals['current_price']}
- Hold Minutes: {signals['hold_minutes']}

Data interval: {interval}, lookback: {duration}
{news_summary}
"""

                with st.spinner("Gemini is analyzing..."):
                    response = model.generate_content(prompt)

                progress_bar.progress(100)
                status_text.success("Analysis complete!")

                # Display results
                st.subheader("🎯 Trade Suggestion")
                st.markdown("---")
                st.success(response.text)

                with st.expander("📖 Full AI Response"):
                    st.markdown(response.text)

                # Download button
                st.download_button(
                    "💾 Download Analysis",
                    response.text,
                    file_name=f"{symbol.upper()}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            progress_bar.empty()