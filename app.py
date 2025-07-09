# 📊 Professional Streamlit Stock Dashboard with Advanced Features

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date
import requests
import feedparser

# --- Page Config --- mine 
st.set_page_config(page_title="📈 Advanced Stock Dashboard", layout="wide")

# --- Format Helper ---
def format_large_number(num):
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    if num > 1e12:
        return f"{num / 1e12:.2f}T"
    if num > 1e9:
        return f"{num / 1e9:.2f}B"
    if num > 1e6:
        return f"{num / 1e6:.2f}M"
    return f"{num:.2f}"

# --- Sidebar ---
st.sidebar.header("⚙️ Filter & Display Options")
all_tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "AMZN", "META"]
selected_tickers = st.sidebar.multiselect("Select Stock Ticker(s):", all_tickers, default=["AAPL"])
start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# --- Cache data fetching ---
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end, auto_adjust=True)

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    logo_fallbacks = {
        "AAPL": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
        "MSFT": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
        "GOOGL": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
        "TSLA": "https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png",
        "NVDA": "https://upload.wikimedia.org/wikipedia/en/6/6c/Nvidia_logo.svg",
        "AMZN": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        "META": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Meta-Logo.png",
    }

    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        fast_info = tkr.fast_info

        name = info.get("longName") or info.get("shortName") or ticker
        logo = info.get("logo_url", logo_fallbacks.get(ticker))

        financials = {
            "marketCap": fast_info.get("market_cap") or info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "dividendYield": info.get("dividendYield"),
            "fiftyTwoWeekHigh": fast_info.get("year_high") or info.get("fiftyTwoWeekHigh"),
        }

        return financials, name, logo
    except:
        return {
            "marketCap": None,
            "peRatio": None,
            "dividendYield": None,
            "fiftyTwoWeekHigh": None
        }, ticker, logo_fallbacks.get(ticker)

@st.cache_data(ttl=3600)
def get_rss_news(ticker):
    feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
    return feed.entries[:5]

# --- Main App ---
if not selected_tickers:
    st.warning("Please select at least one stock ticker.")
    st.stop()

df_multi = load_data(selected_tickers, start_date, end_date)
print (df_multi)

if len(selected_tickers) > 1:
    st.title("📊 Stock Price Comparison")
    close_prices = df_multi['Close']
    normalized = (close_prices / close_prices.iloc[0]) * 100
    fig = go.Figure()
    for ticker in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], name=ticker))
    fig.update_layout(title="Normalized Stock Price Performance (Base 100)", height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    ticker = selected_tickers[0]
    df = df_multi.xs(ticker, level=1, axis=1) if isinstance(df_multi.columns, pd.MultiIndex) else df_multi

    info, name, logo = get_company_info(ticker)
    col1, col2 = st.columns([1, 4])
    with col1:
        if logo:
            st.image(logo, width=100)
    with col2:
        st.title(f"{name} ({ticker})")

    st.subheader("📌 Key Financial Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Cap", format_large_number(info.get("marketCap")))
    m2.metric("P/E Ratio", f"{info.get('peRatio'):.2f}" if info.get("peRatio") else "N/A")
    m3.metric("Dividend Yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "N/A")
    m4.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh'):.2f}" if info.get("fiftyTwoWeekHigh") else "N/A")

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + 2 * std
    df['Lower_BB'] = df['MA20'] - 2 * std

    # --- Candlestick Chart ---
    st.subheader("📊 Price Chart")
    candlestick = go.Figure()
    candlestick.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlesticks'))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='orange')))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], name='Upper BB', line=dict(color='gray', dash='dash')))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], name='Lower BB', line=dict(color='gray', dash='dash')))
    candlestick.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(candlestick, use_container_width=True)

    # --- Volume ---
    st.subheader("📦 Volume")
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='gray'))
    vol_fig.update_layout(height=200)
    st.plotly_chart(vol_fig, use_container_width=True)

    # --- RSI and MACD ---
    st.subheader("📐 RSI and MACD")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(height=250)
    st.plotly_chart(rsi_fig, use_container_width=True)

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='navy')))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='skyblue')))
    macd_fig.update_layout(height=250)
    st.plotly_chart(macd_fig, use_container_width=True)

    # --- News ---
    st.subheader("📰 Live News")
    try:
        news_items = get_rss_news(ticker)
        for item in news_items:
            st.markdown(f"• [{item.title}]({item.link})  \n*{item.published}*")
    except:
        st.info("No live news available.")

    # --- Download ---
    st.subheader("📥 Download CSV")
    csv = df.to_csv().encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")
