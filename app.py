# 📊 Professional Streamlit Stock Dashboard with Advanced Features

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date, datetime, timedelta
import requests
import feedparser
import json
from pathlib import Path

# --- Page & Caching Config ---
st.set_page_config(page_title="📈 Robust Stock Dashboard", layout="wide")

# --- Constants & Setup ---
DATA_DIR = Path("data")
PRICE_CACHE_TTL = timedelta(hours=1)
INFO_CACHE_TTL = timedelta(days=1)
NEWS_CACHE_TTL = timedelta(minutes=30)

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Format & Helper Functions ---
def format_large_number(num):
    """Formats a large number into a human-readable string (e.g., 1.23T, 4.56B, 7.89M)."""
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    if abs(num) > 1e12:
        return f"{num / 1e12:.2f}T"
    if abs(num) > 1e9:
        return f"{num / 1e9:.2f}B"
    if abs(num) > 1e6:
        return f"{num / 1e6:.2f}M"
    return f"{num:,.2f}"

def is_cache_valid(file_path, ttl):
    """Checks if a cached file is still valid based on its modification time and TTL."""
    if not file_path.exists():
        return False
    file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    return datetime.now() - file_mod_time < ttl

# --- Data Loading with Robust Caching ---
def load_price_data(ticker):
    """
    Loads historical price data for a ticker.
    First, tries to load from a valid cache file. If that fails or the cache
    is stale, it fetches fresh data from yfinance and updates the cache.
    """
    cache_file = DATA_DIR / f"{ticker}_price_data.csv"
    
    if is_cache_valid(cache_file, PRICE_CACHE_TTL):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        data = yf.download(ticker, period="max", auto_adjust=True)
        if data.empty:
            st.error(f"Could not download price data for {ticker}. It might be delisted or an invalid ticker.")
            return None
        data.to_csv(cache_file)
        return data
    except Exception as e:
        st.error(f"Failed to fetch price data for {ticker}: {e}")
        if cache_file.exists():
            st.warning(f"Using stale price data for {ticker} due to API error.")
            try:
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None

def load_company_info(ticker):
    """
    Loads company information (metadata, financials).
    Uses a local JSON file for caching.
    """
    cache_file = DATA_DIR / f"{ticker}_info.json"
    logo_fallbacks = {
        "AAPL": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
        "MSFT": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
        "GOOGL": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
        "TSLA": "https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png",
        "NVDA": "https://upload.wikimedia.org/wikipedia/en/6/6c/Nvidia_logo.svg",
        "AMZN": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        "META": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Meta-Logo.png",
    }
    default_info = {
        "longName": ticker, "logo_url": logo_fallbacks.get(ticker),
        "marketCap": None, "trailingPE": None, "dividendYield": None, "fiftyTwoWeekHigh": None
    }

    if is_cache_valid(cache_file, INFO_CACHE_TTL):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass 

    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        filtered_info = {
            "longName": info.get("longName", ticker),
            "logo_url": info.get("logo_url", logo_fallbacks.get(ticker)),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "dividendYield": info.get("dividendYield"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        }
        with open(cache_file, 'w') as f:
            json.dump(filtered_info, f)
        return filtered_info
    except Exception as e:
        st.error(f"Failed to fetch company info for {ticker}: {e}")
        if cache_file.exists():
            st.warning(f"Using stale company info for {ticker} due to API error.")
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return default_info
        return default_info

@st.cache_data(ttl=NEWS_CACHE_TTL)
def get_rss_news(ticker):
    """Fetches news from Yahoo Finance RSS feed."""
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}®ion=US&lang=en-US")
        return feed.entries[:5]
    except Exception as e:
        st.warning(f"Could not fetch news for {ticker}: {e}")
        return []

# --- Sidebar ---
st.sidebar.header("⚙️ Filter & Display Options")
all_tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "AMZN", "META", "JPM"]
selected_tickers = st.sidebar.multiselect("Select Stock Ticker(s):", all_tickers, default=["AAPL"])
start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Data Caching Status:**
- **Price Data:** Refreshed every `{int(PRICE_CACHE_TTL.total_seconds()/3600)}` hour(s).
- **Company Info:** Refreshed every `{int(INFO_CACHE_TTL.total_seconds()/86400)}` day(s).
""")

# --- Main App ---
if not selected_tickers:
    st.warning("Please select at least one stock ticker from the sidebar.")
    st.stop()

all_data = {ticker: load_price_data(ticker) for ticker in selected_tickers}
all_data = {ticker: data for ticker, data in all_data.items() if data is not None}

if not all_data:
    st.error("Could not load data for any of the selected tickers. Please check the ticker symbols or try again later.")
    st.stop()

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

for ticker, df in all_data.items():
    all_data[ticker] = df.loc[start_date_str:end_date_str]


if len(all_data) > 1:
    st.title("📊 Stock Price Comparison")
    close_prices_df = pd.concat({ticker: df['Close'] for ticker, df in all_data.items() if not df.empty}, axis=1)
    
    if close_prices_df.empty:
        st.warning("No data available for the selected date range. Please select an earlier Start Date.")
        st.stop()

    close_prices_df.dropna(inplace=True)
    if close_prices_df.empty:
        st.warning("Inconsistent trading day data for the selected stocks in this date range. Cannot create comparison.")
        st.stop()

    normalized = (close_prices_df / close_prices_df.iloc[0]) * 100
    
    fig = go.Figure()
    for ticker in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], name=ticker, hovertemplate=f'<b>{ticker}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>'))
    fig.update_layout(
        title="Normalized Stock Price Performance (Base 100)",
        height=600,
        legend_title_text='Tickers',
        yaxis_title="Normalized Price"
    )
    st.plotly_chart(fig, use_container_width=True)

else: # Single ticker view
    ticker = list(all_data.keys())[0]
    df = all_data[ticker]

    if df.empty:
        st.warning("No data available for the selected date range. Please select an earlier Start Date.")
        st.stop()
        
    info = load_company_info(ticker)
    name = info.get("longName", ticker)
    logo = info.get("logo_url")

    col1, col2 = st.columns([1, 4])
    with col1:
        if logo:
            st.image(logo, width=100)
    with col2:
        st.title(f"{name} ({ticker})")
        st.subheader("Live Market Data & Analysis")

    st.subheader("📌 Key Financial Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Cap", format_large_number(info.get("marketCap")))
    m2.metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A")
    m3.metric("Dividend Yield", f"{info.get('dividendYield')*100:.2f}" if info.get('dividendYield') else "N/A")
    m4.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh'):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
    st.markdown("---")

    close_prices = df['Close'].squeeze()
    df['MA20'] = close_prices.rolling(window=20).mean()
    df['MA50'] = close_prices.rolling(window=50).mean()
    std_dev = close_prices.rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + (2 * std_dev)
    df['Lower_BB'] = df['MA20'] - (2 * std_dev)
    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = close_prices.ewm(span=12, adjust=False).mean()
    ema26 = close_prices.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    st.subheader("📊 Candlestick Chart with Technical Indicators")
    candlestick = go.Figure()
    candlestick.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-Day MA', line=dict(color='blue', width=1.5)))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-Day MA', line=dict(color='orange', width=1.5)))
    candlestick.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], name='Upper Bollinger Band', line=dict(color='rgba(128,128,128,0.5)', dash='dash')))
    
    # ==============================================================================
    # --- SYNTAX FIX: Moved `fill` and `fillcolor` INSIDE the go.Scatter() call ---
    # ==============================================================================
    candlestick.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower_BB'],
        name='Lower Bollinger Band',
        line=dict(color='rgba(128,128,128,0.5)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)'
    ))
    # ==============================================================================

    candlestick.update_layout(xaxis_rangeslider_visible=False, height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(candlestick, use_container_width=True)
    
    st.subheader("📐 Technical Indicator Plots")
    tab1, tab2, tab3 = st.tabs(["Volume", "RSI (Relative Strength Index)", "MACD (Moving Average Convergence Divergence)"])

    with tab1:
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='gray'))
        vol_fig.update_layout(height=300, title_text="Trading Volume")
        st.plotly_chart(vol_fig, use_container_width=True)

    with tab2:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="bottom right")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right")
        rsi_fig.update_layout(height=300, yaxis_range=[0,100], title_text="RSI")
        st.plotly_chart(rsi_fig, use_container_width=True)

    with tab3:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='navy')))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal Line', line=dict(color='skyblue')))
        macd_fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'], name='Histogram', marker_color='rgba(128,128,128,0.5)'))
        macd_fig.update_layout(height=300, title_text="MACD")
        st.plotly_chart(macd_fig, use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns([3, 1])
    with col3:
        st.subheader("📰 Live News")
        news_items = get_rss_news(ticker)
        if news_items:
            for item in news_items:
                st.markdown(f"• [{item.title}]({item.link})  \n  *<small>{item.published}</small>*", unsafe_allow_html=True)
        else:
            st.info("No live news available for this ticker.")
            
    with col4:
        st.subheader("📥 Download Data")
        csv = df.to_csv().encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{ticker}_data_{start_date_str}_to_{end_date_str}.csv",
            mime="text/csv",
            use_container_width=True
        )