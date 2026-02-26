import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM

st.set_page_config(page_title="Stock Probabilities with HMM", layout="wide")
st.title("📈 Stock Buy/Hold/Sell Probabilities with HMM")

# --- Fetch Data ---
@st.cache_data
def get_market_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        data['RSI'] = compute_rsi(data['Close'])
        data['MA50'] = data['Close'].rolling(50, min_periods=1).mean()
        data['MA200'] = data['Close'].rolling(200, min_periods=1).mean()
        data['VOL_MA'] = data['Volume'].rolling(20, min_periods=1).mean()
        # Fill initial NaNs with first valid value
        data.fillna(method='bfill', inplace=True)
        return data
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral if not enough data

# --- Compute Scores ---
def compute_score(df):
    score = np.zeros(len(df))
    score += np.where(df['RSI'] < 30, 1.0, 0)
    score += np.where(df['RSI'] > 70, -1.0, 0)
    score += np.where((df['Close'] > df['MA50']) & (df['MA50'] > df['MA200']), 0.8, 0)
    score += np.where((df['Close'] < df['MA50']) & (df['MA50'] < df['MA200']), -0.8, 0)
    score += np.where((df['Close'] > df['MA50']) & (df['MA50'] <= df['MA200']), 0.4, 0)
    score += np.where((df['Close'] <= df['MA50']) & (df['MA50'] >= df['MA200']), -0.4, 0)
    score += np.where(df['Volume'] > df['VOL_MA'] * 1.5, 0.3, 0)
    return np.clip(score, -1.5, 1.5)

def score_to_probabilities(score):
    buy = 1 / (1 + np.exp(-4 * (score - 0.1)))
    sell = 1 / (1 + np.exp(-4 * (-score - 0.1)))
    hold = 1 - buy - sell
    probs = np.array([buy, hold, sell])
    probs = np.clip(probs, 0, 1)
    probs /= probs.sum()
    return probs

# --- HMM ---
def compute_hmm(df, n_states=3):
    returns = df['Close'].pct_change().fillna(0).values.reshape(-1,1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(returns)
    hidden_states = model.predict(returns)
    prob_matrix = model.predict_proba(returns)
    return hidden_states, prob_matrix

# --- Streamlit UI ---
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker Symbol", value="NBIS").upper()
with col2:
    today = dt.date.today()
    start_date = st.date_input("Start Date", value=today - dt.timedelta(days=365))
    end_date = st.date_input("End Date", value=today)

if st.button("🚀 Compute Probabilities"):
    data = get_market_data(ticker, start_date, end_date)
    if data is None or data.empty:
        st.error(f"No data found for {ticker}")
    else:
        # Retrospective and predicted probabilities
        data['Score'] = compute_score(data)
        # Predicted: shift 1 day forward
        data['Score_Pred'] = data['Score'].shift(1).fillna(method='bfill')
        probs_ret = np.array([score_to_probabilities(s) for s in data['Score']])
        probs_pred = np.array([score_to_probabilities(s) for s in data['Score_Pred']])
        data[['Buy_R', 'Hold_R', 'Sell_R']] = probs_ret
        data[['Buy_P', 'Hold_P', 'Sell_P']] = probs_pred

        # HMM regimes
        hidden_states, prob_matrix = compute_hmm(data)
        data['HMM_State'] = hidden_states
        prob_df = pd.DataFrame(prob_matrix, columns=[f"Regime_{i}" for i in range(prob_matrix.shape[1])], index=data.index)
        # Map HMM to B/H/S probabilities (simple mapping: highest prob state -> Buy/Sell/Hold)
        data['Buy_HMM'] = prob_df.idxmax(axis=1).map({'Regime_0':0.7,'Regime_1':0.2,'Regime_2':0.1})
        data['Hold_HMM'] = prob_df.idxmax(axis=1).map({'Regime_0':0.2,'Regime_1':0.7,'Regime_2':0.1})
        data['Sell_HMM'] = prob_df.idxmax(axis=1).map({'Regime_0':0.1,'Regime_1':0.1,'Regime_2':0.8})

        # Slider for HMM shift
        shift_days = st.slider("Shift HMM by days", min_value=0, max_value=14, value=0)
        shifted_prob_df = prob_df.shift(shift_days).fillna(method='bfill')
        data['Buy_HMM_Shift'] = shifted_prob_df.idxmax(axis=1).map({'Regime_0':0.7,'Regime_1':0.2,'Regime_2':0.1})
        data['Hold_HMM_Shift'] = shifted_prob_df.idxmax(axis=1).map({'Regime_0':0.2,'Regime_1':0.7,'Regime_2':0.1})
        data['Sell_HMM_Shift'] = shifted_prob_df.idxmax(axis=1).map({'Regime_0':0.1,'Regime_1':0.1,'Regime_2':0.8})

        # --- Chart ---
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Retrospective Probabilities","Predicted Probabilities","HMM Regime Probabilities"),
            row_heights=[0.33,0.33,0.34]
        )

        # Retrospective
        fig.add_trace(go.Scatter(x=data.index, y=data['Buy_R'], stackgroup='one', name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Hold_R'], stackgroup='one', name='Hold'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Sell_R'], stackgroup='one', name='Sell'), row=1, col=1)
        # Predicted
        fig.add_trace(go.Scatter(x=data.index, y=data['Buy_P'], stackgroup='one', name='Buy'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Hold_P'], stackgroup='one', name='Hold'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Sell_P'], stackgroup='one', name='Sell'), row=2, col=1)
        # HMM Shifted
        fig.add_trace(go.Scatter(x=data.index, y=data['Buy_HMM_Shift'], stackgroup='one', name='Buy'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Hold_HMM_Shift'], stackgroup='one', name='Hold'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Sell_HMM_Shift'], stackgroup='one', name='Sell'), row=3, col=1)

        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- Table ---
        st.subheader("Recent Probabilities & HMM Regimes")
        st.dataframe(
            data[['Close','Volume','Buy_R','Hold_R','Sell_R','Buy_P','Hold_P','Sell_P',
                  'Buy_HMM_Shift','Hold_HMM_Shift','Sell_HMM_Shift']].tail(10).round(3),
            use_container_width=True
        )

