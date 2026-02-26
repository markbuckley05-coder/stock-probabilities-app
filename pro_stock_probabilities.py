import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Stock Probabilities", layout="wide")
st.title("📈 Market-Implied Buy / Hold / Sell Probabilities")

# -----------------------------
# DATA DOWNLOAD (WITH WARM-UP)
# -----------------------------
@st.cache_data
def get_market_data(ticker, start, end):
    """
    Fetch historical stock data with extra history for indicator warm-up
    """
    extended_start = start - dt.timedelta(days=400)  # extra days for MA200 warm-up
    df = yf.download(ticker, start=extended_start, end=end, progress=False)

    if df.empty:
        return None

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str)

    # Store user-selected start date
    df.attrs["user_start"] = start
    return df

# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------
def compute_technical_signals(df):
    df = df.copy()

    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("Yahoo data missing required columns")

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving averages
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["VOL_MA"] = df["Volume"].rolling(20).mean()

    # Drop rows where indicators cannot be computed
    df = df.dropna(subset=["RSI", "MA50", "MA200", "VOL_MA"])
    return df

# -----------------------------
# FUNDAMENTALS
# -----------------------------
@st.cache_data
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe": info.get("trailingPE", np.nan),
            "growth": info.get("revenueGrowth", np.nan)
        }
    except:
        return {"pe": np.nan, "growth": np.nan}

# -----------------------------
# SCORE ENGINE
# -----------------------------
def compute_score(df, fundamentals=None, analyst_score=0, macro_score=0):
    score = pd.Series(0.0, index=df.index)

    # RSI
    score += np.where(df["RSI"] < 30, 1.0, 0)
    score += np.where(df["RSI"] > 70, -1.0, 0)

    # Price vs MAs
    score += np.where((df["Close"] > df["MA50"]) & (df["MA50"] > df["MA200"]), 0.8, 0)
    score += np.where((df["Close"] < df["MA50"]) & (df["MA50"] < df["MA200"]), -0.8, 0)

    # Volume spike
    score += np.where(df["Volume"] > df["VOL_MA"] * 1.5, 0.3, 0)

    # Fundamentals
    if fundamentals:
        if pd.notna(fundamentals["growth"]) and fundamentals["growth"] > 0.2:
            score += 0.3
        if pd.notna(fundamentals["pe"]) and fundamentals["pe"] > 80:
            score -= 0.2

    # Analyst + Macro input
    score += analyst_score
    score += macro_score

    return score.clip(-1.5, 1.5)

# -----------------------------
# SCORE → PROBABILITIES
# -----------------------------
def score_to_probabilities(score_series):
    buy = 1 / (1 + np.exp(-4 * (score_series - 0.1)))
    sell = 1 / (1 + np.exp(-4 * (-score_series - 0.1)))
    hold = 1 - buy - sell

    probs = np.vstack([buy, hold, sell]).T
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return pd.DataFrame(probs, index=score_series.index,
                        columns=["Buy", "Hold", "Sell"])

# -----------------------------
# USER INPUTS
# -----------------------------
col1, col2 = st.columns([2,1])

with col1:
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()

with col2:
    today = dt.date.today()
    start = st.date_input("Start Date", today - dt.timedelta(days=365))
    end = st.date_input("End Date", today)

st.markdown("### Optional Sentiment Inputs")
analyst_score_input = st.slider("Analyst Sentiment", -0.5, 0.5, 0.1, 0.05)
macro_score_input = st.slider("Macro Environment", -0.5, 0.5, 0.0, 0.05)

# -----------------------------
# RUN MODEL
# -----------------------------
if st.button("🚀 Compute Probabilities", type="primary"):

    data = get_market_data(ticker, start, end)

    if data is None:
        st.error("No data found for ticker.")
        st.stop()

    data = compute_technical_signals(data)

    # -----------------------------
    # Trim to user-selected window AFTER indicators ready
    # Convert user_start to datetime64 to fix comparison error
    user_start = pd.to_datetime(data.attrs["user_start"])
    data = data.loc[data.index >= user_start]

    fundamentals = get_fundamentals(ticker)

    # Compute score & probabilities
    data["Score"] = compute_score(
        data,
        fundamentals=fundamentals,
        analyst_score=analyst_score_input,
        macro_score=macro_score_input
    )

    probs = score_to_probabilities(data["Score"])
    data = pd.concat([data, probs], axis=1)

    # -----------------------------
    # Display table
    st.subheader("Recent Probabilities")
    st.dataframe(data[["Buy","Hold","Sell"]].tail(10).round(3))

    # -----------------------------
    # Plot probabilities + price/volume
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Probability Model", "Price & Volume"),
        row_heights=[0.6,0.4]
    )

    fig.add_trace(go.Scatter(x=data.index, y=data["Buy"], stackgroup="one", name="Buy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Hold"], stackgroup="one", name="Hold"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Sell"], stackgroup="one", name="Sell"), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price"), row=2, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"]/1e6, name="Volume (M)"), row=2, col=1)

    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
