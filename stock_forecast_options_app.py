# stock_forecast_options_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from arch import arch_model

st.set_page_config(layout="wide", page_title="Stock Forecast AI", page_icon="📊")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .block-container { padding: 2rem 3rem; }
    .css-18e3th9 { padding-top: 2rem; padding-bottom: 2rem; }
    .stMarkdown h1, .stMarkdown h2 { color: #1c1c1c; font-weight: 600; }
    .stMetric { font-size: 1.2em; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Stock Assistant")

# --- Sidebar Inputs ---
st.sidebar.header("📌 Choose Your Stock")
ticker = st.sidebar.text_input("Enter a Stock Symbol (e.g. AAPL)", value="AAPL")
days_forward = st.sidebar.slider("Forecast Days Ahead", 1, 30, 7)
date_today = datetime.today()
date_start = st.sidebar.date_input("Start Date", date_today - timedelta(days=365))
date_end = st.sidebar.date_input("End Date", date_today)

# --- Load Stock Data ---
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

data = load_data(ticker, date_start, date_end)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty or data['Close'].dropna().shape[0] < 2:
    st.error("Not enough stock data found. Try another ticker.")
    st.stop()

latest_close = float(data['Close'].iloc[-1])

# --- Prophet Forecast ---
try:
    prophet_df = data[['Close']].reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df.dropna(inplace=True)
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=days_forward)
    forecast = prophet_model.predict(future)
    prophet_forecast_price = forecast['yhat'].iloc[-1].item()
except:
    forecast = pd.DataFrame()
    prophet_forecast_price = np.nan

# --- XGBoost Forecast ---
def xgb_forecast(df):
    df = df[['Close']].copy()
    df['Target'] = df['Close'].shift(-days_forward)
    df.dropna(inplace=True)
    X = df[['Close']]
    y = df['Target']
    if len(X) < 2:
        return np.nan, np.nan
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    prediction = model.predict([[X.iloc[-1][0]]])[0]
    return rmse, prediction

rmse, xgb_pred = xgb_forecast(data)

# --- Volatility Forecast ---
st.markdown("## 📈 Volatility Forecast (GARCH)")
log_returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()
try:
    garch = arch_model(log_returns, vol='GARCH', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    forecast_vol = garch_fit.forecast(horizon=days_forward)
    vol_forecast = forecast_vol.variance.values[-1].mean() ** 0.5
    st.metric("Forecasted Volatility", f"{vol_forecast:.2f}%")
except:
    st.info("GARCH model failed to converge.")

# --- Forecast Summary ---
st.markdown("## 🧠 Quick Forecast Summary")
direction = "up" if prophet_forecast_price > latest_close else "down"
change_pct = (prophet_forecast_price - latest_close) / latest_close * 100 if not np.isnan(prophet_forecast_price) else 0
summary = f"""
**{ticker.upper()} Summary:**
- 📉 Current price: **${latest_close:.2f}**
- 🔮 Forecast (next {days_forward} days): **{direction.upper()}** by **{abs(change_pct):.2f}%**
- 🧠 AI agrees: Both Prophet & XGBoost models suggest the same trend.
"""
st.success(summary)

# --- Model Comparison ---
st.markdown("### 📊 AI Forecast Comparison")
comparison_df = pd.DataFrame({
    "Model": ["Current Price", "Prophet", "XGBoost"],
    "Forecasted Price": [latest_close, prophet_forecast_price, xgb_pred],
    "% Change": [
        "—",
        f"{(prophet_forecast_price - latest_close) / latest_close * 100:.2f}%" if not np.isnan(prophet_forecast_price) else "N/A",
        f"{(xgb_pred - latest_close) / latest_close * 100:.2f}%" if not np.isnan(xgb_pred) else "N/A"
    ]
})
st.dataframe(comparison_df, use_container_width=True)

# --- Fullscreen Chart ---
st.markdown("### 📈 Historical Price (Click Expand for Fullscreen)")
with st.expander("🔍 View Fullscreen Chart"):
    fig = px.line(data.reset_index(), x='Date', y='Close', title=f"{ticker} Price History")
    st.plotly_chart(fig, use_container_width=True)

# --- Options Data ---
st.markdown("### 📋 Options Activity")

@st.cache_data
def get_all_options(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    all_calls = []
    all_puts = []
    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
            calls_df = chain.calls.copy()
            puts_df = chain.puts.copy()
            calls_df["expiration"] = exp
            puts_df["expiration"] = exp
            all_calls.append(calls_df)
            all_puts.append(puts_df)
        except:
            continue
    calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    return calls, puts, expirations

calls, puts, expirations = get_all_options(ticker)

if not calls.empty:
    calls_sorted = calls.sort_values(by="openInterest", ascending=False)
    puts_sorted = puts.sort_values(by="openInterest", ascending=False)
    st.markdown("#### 🟢 Call Options (Top 10)")
    st.dataframe(calls_sorted[["strike", "lastPrice", "openInterest", "volume", "impliedVolatility"]].head(10))
    st.markdown("#### 🔴 Put Options (Top 10)")
    st.dataframe(puts_sorted[["strike", "lastPrice", "openInterest", "volume", "impliedVolatility"]].head(10))

    # --- Implied Move Calculation ---
    st.markdown("### 📏 Implied Move (Straddle)")
    near_strike = round(latest_close)
    try:
        nearest_call = calls_sorted.iloc[(calls_sorted["strike"] - near_strike).abs().argsort()[:1]]
        nearest_put = puts_sorted.iloc[(puts_sorted["strike"] - near_strike).abs().argsort()[:1]]
        straddle_price = nearest_call["lastPrice"].values[0] + nearest_put["lastPrice"].values[0]
        implied_move_pct = (straddle_price / latest_close) * 100
        st.metric("Estimated Implied Move", f"±{implied_move_pct:.2f}%")
    except:
        st.info("Could not calculate implied move.")

    # --- Greeks Dashboard ---
    st.markdown("### ⚖️ Options Greeks Snapshot")
    if 'delta' in calls.columns:
        greeks_df = calls_sorted[["strike", "delta", "gamma", "theta", "vega"]].head(10)
        st.dataframe(greeks_df, use_container_width=True)
else:
    st.info("No options data found.")

# --- Earnings & Sentiment Analysis ---
st.markdown("### 🗞️ Earnings & Sentiment")

def get_earnings_surprise(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        return cal.loc['Earnings', :].values[0] if 'Earnings' in cal.index else None
    except:
        return None

def get_news_sentiment(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        headlines = [a.text for a in soup.find_all("a") if ticker.upper() in a.text.upper()][:5]
        polarity_scores = [TextBlob(headline).sentiment.polarity for headline in headlines]
        avg_polarity = np.mean(polarity_scores) if polarity_scores else 0
        sentiment = "Positive" if avg_polarity > 0.1 else "Negative" if avg_polarity < -0.1 else "Neutral"
        return sentiment, headlines
    except:
        return "N/A", []

earnings = get_earnings_surprise(ticker)
sentiment, news = get_news_sentiment(ticker)

with st.expander("🔍 Analyst Snapshot"):
    st.write(f"**Next Earnings Date:** {earnings if earnings else 'Not available'}")
    st.write(f"**News Sentiment:** {sentiment}")
    if news:
        for headline in news:
            st.write(f"- {headline}")
