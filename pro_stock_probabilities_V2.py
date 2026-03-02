import datetime as dt
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---- Streamlit config: MUST be the first Streamlit command ----
st.set_page_config(page_title="Stock Probabilities v2 (Horizon + Validation)", layout="wide")
st.title("📈 Stock Probabilities v2 — Horizon Probabilities + Walk-Forward Validation")
st.caption("This version adds horizon-aware Rise/Stay/Fall probabilities (sum=1), walk-forward validation, and an optional HMM panel.")

warnings.filterwarnings("ignore", message="Model is not converging*")


# ------------------------
# Helpers
# ------------------------
def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _row_normalize(df: pd.DataFrame) -> pd.DataFrame:
    s = df.sum(axis=1).replace(0, np.nan)
    out = df.div(s, axis=0).fillna(0.0)
    return out


@st.cache_data(show_spinner=False)
def get_market_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    """
    Fetch OHLCV from yfinance and compute indicators.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if df is None or df.empty:
            return None

        # yfinance sometimes returns MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Clean index (Plotly-friendly)
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return None

        # Indicators (kept for reference + future features)
        df["RSI"] = compute_rsi(df["Close"])
        df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
        df["MA200"] = df["Close"].rolling(200, min_periods=1).mean()
        df["VOL_MA"] = df["Volume"].rolling(20, min_periods=1).mean()

        df = df.bfill()
        return df

    except Exception as e:
        st.error(f"Data fetch error for '{ticker}': {e}")
        return None


def compute_score(df: pd.DataFrame) -> np.ndarray:
    """
    Your existing rule-based score.
    This is our v1 feature for the probability engine.
    """
    score = np.zeros(len(df), dtype=float)

    score += np.where(df["RSI"] < 30, 1.0, 0.0)
    score += np.where(df["RSI"] > 70, -1.0, 0.0)

    score += np.where((df["Close"] > df["MA50"]) & (df["MA50"] > df["MA200"]), 0.8, 0.0)
    score += np.where((df["Close"] < df["MA50"]) & (df["MA50"] < df["MA200"]), -0.8, 0.0)

    score += np.where((df["Close"] > df["MA50"]) & (df["MA50"] <= df["MA200"]), 0.4, 0.0)
    score += np.where((df["Close"] <= df["MA50"]) & (df["MA50"] >= df["MA200"]), -0.4, 0.0)

    score += np.where(df["Volume"] > df["VOL_MA"] * 1.5, 0.3, 0.0)

    return np.clip(score, -1.5, 1.5)


def compute_hmm(df: pd.DataFrame, n_states: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Gaussian HMM to standardized daily returns. (Optional display only in v2.)
    """
    returns = df["Close"].pct_change().fillna(0.0).to_numpy()
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    X = ((returns - mu) / sigma).reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X)
    hidden_states = model.predict(X)
    prob_matrix = model.predict_proba(X)
    return hidden_states, prob_matrix


def compute_vol_scaled_labels(
    df: pd.DataFrame,
    horizon: int,
    lookback_L: int = 20,
    k: float = 0.35,
    sigma_floor: float = 0.002,   # 0.2% daily
    delta_floor: float = 0.003,   # 0.3% over horizon
) -> pd.DataFrame:
    """
    Computes:
      - daily returns r_t
      - rolling sigma_t with floor
      - delta_{t,H} = max(k*sigma_t*sqrt(H), delta_floor)
      - forward return R_{t,H}
      - label in {Rise, Stay, Fall} based on +/- delta_{t,H}

    IMPORTANT: Uses only information available at time t for sigma_t and delta_{t,H}.
    """
    out = df.copy()

    out["r1"] = out["Close"].pct_change()
    out["sigma_t"] = out["r1"].rolling(lookback_L, min_periods=lookback_L).std()
    out["sigma_t"] = out["sigma_t"].fillna(np.nan)
    out["sigma_t"] = out["sigma_t"].clip(lower=sigma_floor)

    out["sigma_t_H"] = out["sigma_t"] * np.sqrt(horizon)
    out["delta_t_H"] = (k * out["sigma_t_H"]).clip(lower=delta_floor)

    out["FwdRet_H"] = out["Close"].shift(-horizon) / out["Close"] - 1.0

    # Label (only valid where FwdRet_H exists)
    out["Y"] = np.where(
        out["FwdRet_H"] > out["delta_t_H"], "Rise",
        np.where(out["FwdRet_H"] < -out["delta_t_H"], "Fall", "Stay")
    )

    # Where we don't have future data, label is unknown
    out.loc[out["FwdRet_H"].isna(), "Y"] = np.nan

    return out


def _safe_log_loss(probs: np.ndarray, y_true_idx: np.ndarray, eps: float = 1e-12) -> float:
    """
    Multi-class log loss. probs shape (n,3), y_true_idx shape (n,)
    """
    probs = np.clip(probs, eps, 1.0 - eps)
    p = probs[np.arange(len(y_true_idx)), y_true_idx]
    return float(-np.mean(np.log(p)))


# ============================================================
# ✅ BLOCK 1: BIN walk-forward now supports rolling window
# ============================================================
def walk_forward_probabilities_from_score_bins(
    df: pd.DataFrame,
    score_col: str = "Score",
    y_col: str = "Y",
    n_bins: int = 15,
    min_train: int = 252,
    step: int = 20,
    train_window: int | None = None,  # ✅ NEW: rolling window length (rows). None = expanding.
) -> pd.DataFrame:
    """
    Walk-forward estimation of P(Rise/Stay/Fall) based on Score bins.

    Modes:
      - Expanding (default): train uses all eligible rows from start to fold start
      - Rolling: if train_window is set (e.g. 180), train uses only the last train_window eligible rows

    Returns a DataFrame with columns:
      P_Rise, P_Stay, P_Fall
    aligned to df.index, with NaNs where not predicted.
    """
    work = df[[score_col, y_col]].copy()
    eligible = work.dropna(subset=[score_col, y_col]).copy()

    if len(eligible) < (min_train + step):
        return pd.DataFrame(index=df.index, columns=["P_Rise", "P_Stay", "P_Fall"], dtype=float)

    classes = ["Rise", "Stay", "Fall"]
    probs_out = pd.DataFrame(index=df.index, columns=["P_Rise", "P_Stay", "P_Fall"], dtype=float)

    eligible_idx = eligible.index.to_list()

    start = min_train
    while start < len(eligible_idx):
        # ✅ TRAIN INDEX SELECTION: expanding vs rolling
        if train_window is None:
            train_idx = eligible_idx[:start]
        else:
            left = max(0, start - int(train_window))
            train_idx = eligible_idx[left:start]

        test_idx = eligible_idx[start:start + step]
        if len(test_idx) == 0:
            break

        train = eligible.loc[train_idx].copy()
        test = eligible.loc[test_idx].copy()

        # Build bin edges on training scores only (prevents leakage)
        try:
            quantiles = np.linspace(0, 1, n_bins + 1)
            edges = train[score_col].quantile(quantiles).to_numpy()
            edges = np.unique(edges)
            if len(edges) < 4:
                edges = np.linspace(train[score_col].min() - 1e-6, train[score_col].max() + 1e-6, 8)
        except Exception:
            edges = np.linspace(train[score_col].min() - 1e-6, train[score_col].max() + 1e-6, 8)

        train_bins = pd.cut(train[score_col], bins=edges, include_lowest=True, duplicates="drop")
        test_bins = pd.cut(test[score_col], bins=edges, include_lowest=True, duplicates="drop")

        tab = (
            pd.DataFrame({"bin": train_bins, "y": train[y_col]})
            .groupby("bin")["y"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        for c in classes:
            if c not in tab.columns:
                tab[c] = 0.0
        tab = tab[classes]

        # Global fallback (TRAIN) if test bin unseen
        global_probs = (
            train[y_col]
            .value_counts(normalize=True)
            .reindex(classes)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        global_probs = global_probs / np.clip(global_probs.sum(), 1e-12, None)

        p_test = np.zeros((len(test), 3), dtype=float)
        for i, b in enumerate(test_bins):
            if b in tab.index:
                p_test[i, :] = tab.loc[b, classes].to_numpy(dtype=float)
            else:
                p_test[i, :] = global_probs

        p_test = p_test / np.clip(p_test.sum(axis=1, keepdims=True), 1e-12, None)
        probs_out.loc[test.index, ["P_Rise", "P_Stay", "P_Fall"]] = p_test

        start += step

    return probs_out


def fit_full_mapping_for_live_prediction(
    df: pd.DataFrame,
    score_col: str = "Score",
    y_col: str = "Y",
    n_bins: int = 15,
):
    """
    Fit ScoreBin -> class probabilities on all available labeled data.
    Used ONLY for the latest 'live' prediction where outcome is unknown.
    Returns (edges, tab, global_probs).
    """
    work = df[[score_col, y_col]].dropna(subset=[score_col, y_col]).copy()
    classes = ["Rise", "Stay", "Fall"]

    if len(work) < 50:
        edges = np.linspace(df[score_col].min() - 1e-6, df[score_col].max() + 1e-6, 8)
        global_probs = np.array([1/3, 1/3, 1/3], dtype=float)
        tab = pd.DataFrame(index=pd.IntervalIndex([]), columns=classes).fillna(0.0)
        return edges, tab, global_probs

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = work[score_col].quantile(quantiles).to_numpy()
    edges = np.unique(edges)
    if len(edges) < 4:
        edges = np.linspace(work[score_col].min() - 1e-6, work[score_col].max() + 1e-6, 8)

    # ✅ FIX: bins must always be defined (not only inside the len(edges)<4 branch)
    bins = pd.cut(work[score_col], bins=edges, include_lowest=True, duplicates="drop")

    tab = (
        pd.DataFrame({"bin": bins, "y": work[y_col]})
        .groupby("bin")["y"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    for c in classes:
        if c not in tab.columns:
            tab[c] = 0.0
    tab = tab[classes]

    global_probs = (
        work[y_col]
        .value_counts(normalize=True)
        .reindex(classes)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    global_probs = global_probs / np.clip(global_probs.sum(), 1e-12, None)

    return edges, tab, global_probs


def predict_one_from_mapping(score_value: float, edges, tab: pd.DataFrame, global_probs: np.ndarray) -> np.ndarray:
    """
    Map a single score to probabilities using fitted bins.
    """
    classes = ["Rise", "Stay", "Fall"]
    b = pd.cut(
        pd.Series([score_value]),
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).iloc[0]

    if b in tab.index:
        p = tab.loc[b, classes].to_numpy(dtype=float)
    else:
        p = global_probs

    p = p / np.clip(p.sum(), 1e-12, None)
    return p


# ============================================================
# ✅ BLOCK 2: LOGISTIC walk-forward now supports rolling window
# ============================================================
def walk_forward_probabilities_logistic(
    df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str = "Y",
    min_train: int = 252,
    step: int = 20,
    train_window: int | None = None,  # ✅ NEW: rolling window length (rows). None = expanding.
) -> pd.DataFrame:
    """
    Walk-forward multinomial logistic regression.

    Modes:
      - Expanding (default): train uses all eligible rows from start to fold start
      - Rolling: if train_window is set (e.g. 180), train uses only the last train_window eligible rows

    Returns DataFrame with columns:
      P_Rise, P_Stay, P_Fall
    aligned to df.index.
    """
    probs_out = pd.DataFrame(index=df.index, columns=["P_Rise", "P_Stay", "P_Fall"], dtype=float)

    # Import inside to avoid hard crash if sklearn isn't installed
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        st.error("Logistic engine requires scikit-learn. Install with: pip install scikit-learn")
        st.error(f"scikit-learn import error: {e}")
        return probs_out

    classes = ["Rise", "Stay", "Fall"]

    eligible = df.dropna(subset=feature_cols + [y_col]).copy()
    if len(eligible) < (min_train + step):
        return probs_out

    eligible_idx = eligible.index.to_list()
    start = min_train

    while start < len(eligible_idx):
        # ✅ TRAIN INDEX SELECTION: expanding vs rolling
        if train_window is None:
            train_idx = eligible_idx[:start]
        else:
            left = max(0, start - int(train_window))
            train_idx = eligible_idx[left:start]

        test_idx = eligible_idx[start:start + step]
        if len(test_idx) == 0:
            break

        train = eligible.loc[train_idx].copy()
        test = eligible.loc[test_idx].copy()

        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train[y_col].astype(str).to_numpy()
        X_test = test[feature_cols].to_numpy(dtype=float)

        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            start += step
            continue

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000)),
            ]
        )
        pipe.fit(X_train, y_train)

        p = pipe.predict_proba(X_test)
        model_classes = list(pipe.named_steps["clf"].classes_)

        full = np.zeros((p.shape[0], 3), dtype=float)
        for j, cls in enumerate(classes):
            if cls in model_classes:
                full[:, j] = p[:, model_classes.index(cls)]

        full = full / np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
        probs_out.loc[test.index, ["P_Rise", "P_Stay", "P_Fall"]] = full

        start += step

    # Fill any missing rows with global label frequencies (from eligible only)
    global_probs = (
        eligible[y_col]
        .dropna()
        .astype(str)
        .value_counts(normalize=True)
        .reindex(classes)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    global_probs = global_probs / np.clip(global_probs.sum(), 1e-12, None)

    missing = probs_out.isna().any(axis=1)
    if missing.any():
        probs_out.loc[missing, ["P_Rise", "P_Stay", "P_Fall"]] = global_probs

    return probs_out


def fit_logistic_for_live_prediction(
    df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str = "Y",
):
    """
    Fit logistic on all labeled data (for most recent 'live' probability).
    Returns fitted pipeline or None.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    train = df.dropna(subset=feature_cols + [y_col]).copy()
    if len(train) < 50:
        return None

    X = train[feature_cols].to_numpy(dtype=float)
    y = train[y_col].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return None

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def predict_one_logistic(pipe, x_row: np.ndarray) -> np.ndarray:
    """
    Predict one row using fitted logistic pipe.
    Returns [P(Rise), P(Stay), P(Fall)].
    """
    classes = ["Rise", "Stay", "Fall"]
    p = pipe.predict_proba(x_row.reshape(1, -1))[0]
    model_classes = list(pipe.named_steps["clf"].classes_)

    out = np.zeros(3, dtype=float)
    for j, cls in enumerate(classes):
        if cls in model_classes:
            out[j] = p[model_classes.index(cls)]

    out = out / np.clip(out.sum(), 1e-12, None)
    return out


# ------------------------
# UI
# ------------------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker = st.text_input("Ticker Symbol (Yahoo Finance)", value="AAPL", key="ticker").strip().upper()

with col2:
    today = dt.date.today()
    start_date = st.date_input("Start Date", value=today - dt.timedelta(days=365 * 2), key="start_date")

with col3:
    end_date = st.date_input("End Date", value=today, key="end_date")

st.caption("Tip: some tickers need exchange suffixes (e.g., VOD.L for London).")
horizon = st.select_slider(
    "Prediction horizon (trading days)",
    options=[2, 5, 10, 20],
    value=5,
    key="horizon",
)

# ✅ NEW: Top-level compute toggle (Bin vs Logistic vs Both)
compute_engine = st.radio(
    "Compute engine(s)",
    options=["Both", "Bin", "Logistic"],
    index=0,
    horizontal=True,
    key="compute_engine",
    help="Choose which probability engine(s) to compute. 'Both' computes both and lets you switch in the Display section below.",
)

with st.expander("v1 constants (documented)"):
    st.write("Lookback L = 20 days")
    st.write("k = 0.35")
    st.write("σ_floor = 0.002 (0.2% daily)")
    st.write("δ_floor = 0.003 (0.3% over horizon)")

show_checks = st.checkbox("Show debug checks", value=False, key="show_checks")
show_hmm = st.checkbox("Show HMM panel (optional)", value=False, key="show_hmm")
shift_days = st.slider("HMM shift (days)", min_value=0, max_value=14, value=0, key="shift_days")

y_zoom = st.selectbox(
    "Y-axis scale for probability panel",
    options=["0 to 1 (full probability scale)", "Auto (zoomed)"],
    index=0,
    key="y_zoom",
)

# Walk-forward parameters (kept simple; can be exposed later if needed)
n_bins = 15
min_train = 252
step = 20

# ============================================================
# ✅ BLOCK 3: Rolling window UI control (defines train_window)
# ============================================================
train_window = st.select_slider(
    "Rolling training window (walk-forward)",
    options=[None, 120, 180, 252],
    value=180,
    key="train_window",
    help="None = expanding (old behaviour). 120/180/252 = rolling window length in eligible rows (approx trading days).",
)


def _add_wf_marker(fig: go.Figure, x_date, label: str = "WF predictions start"):
    """
    Safe vertical marker (avoids Plotly add_vline + pandas Timestamp sum issue).
    """
    fig.add_shape(
        type="line",
        x0=x_date,
        x1=x_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(width=1, dash="dash"),
    )
    fig.add_annotation(
        x=x_date,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
    )


if st.button("🚀 Compute v2 (Horizon Probabilities + Validation)", key="compute_btn"):
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        st.stop()

    data = get_market_data(ticker, start_date, end_date)

    if data is None or data.empty:
        st.error(f"No data found for '{ticker}'. Try AAPL/MSFT/SPY, or add an exchange suffix (e.g. VOD.L).")
        st.stop()

    # ------------------------------------------------------------------
    # CRITICAL DIAGNOSTIC: confirm Close is real market data (not 0..N)
    # ------------------------------------------------------------------
    if show_checks:
        st.write("First 10 Close values:")
        st.write(data["Close"].head(10))
        st.write("Last 10 Close values:")
        st.write(data["Close"].tail(10))

    # ------------------------------------------------------------------
    # Force numeric types (prevents Plotly weirdness)
    # ------------------------------------------------------------------
    num_cols = ["Open", "High", "Low", "Close", "Volume", "MA50", "MA200", "RSI"]
    for c in num_cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # Drop rows missing essential OHLC
    data = data.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    if show_checks:
        st.write("OHLC dtypes:", data[["Open", "High", "Low", "Close"]].dtypes.astype(str).to_dict())
        st.write("Close min/max:", float(data["Close"].min()), float(data["Close"].max()))
        st.write("Rows after numeric cleanup:", int(len(data)))

    # ------------------------------------------------------------------
    # Score feature (v1)
    # ------------------------------------------------------------------
    data["Score"] = compute_score(data)

    # ------------------------------------------------------------------
    # Labels for chosen horizon (Rise/Stay/Fall with vol-scaled dead-zone + floors)
    # ------------------------------------------------------------------
    data = compute_vol_scaled_labels(
        data,
        horizon=horizon,
        lookback_L=20,
        k=0.35,
        sigma_floor=0.002,
        delta_floor=0.003,
    )

    # ------------------------------------------------------------------
    # Feature set for Logistic engine (minimal, interpretable, no leakage)
    # ------------------------------------------------------------------
    data["dist_ma50"] = (data["Close"] / data["MA50"]) - 1.0
    data["dist_ma200"] = (data["Close"] / data["MA200"]) - 1.0

    for c in ["sigma_t", "dist_ma50", "dist_ma200"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    feature_cols = ["Score", "RSI", "dist_ma50", "dist_ma200", "sigma_t"]

    # ------------------------------------------------------------------
    # Engine 1: BIN (existing)
    # ------------------------------------------------------------------
    if compute_engine in ("Both", "Bin"):
        probs_bin = walk_forward_probabilities_from_score_bins(
            data,
            score_col="Score",
            y_col="Y",
            n_bins=n_bins,
            min_train=min_train,
            step=step,
            train_window=train_window,
        )
        data["P_Rise_bin"] = pd.to_numeric(probs_bin["P_Rise"], errors="coerce")
        data["P_Stay_bin"] = pd.to_numeric(probs_bin["P_Stay"], errors="coerce")
        data["P_Fall_bin"] = pd.to_numeric(probs_bin["P_Fall"], errors="coerce")
    else:
        # Preserve columns for downstream display code (no functionality loss)
        data["P_Rise_bin"] = np.nan
        data["P_Stay_bin"] = np.nan
        data["P_Fall_bin"] = np.nan

    # ------------------------------------------------------------------
    # Engine 2: LOGISTIC (new)
    # ------------------------------------------------------------------
    if compute_engine in ("Both", "Logistic"):
        probs_log = walk_forward_probabilities_logistic(
            data,
            feature_cols=feature_cols,
            y_col="Y",
            min_train=min_train,
            step=step,
            train_window=train_window,
        )
        data["P_Rise_log"] = pd.to_numeric(probs_log["P_Rise"], errors="coerce")
        data["P_Stay_log"] = pd.to_numeric(probs_log["P_Stay"], errors="coerce")
        data["P_Fall_log"] = pd.to_numeric(probs_log["P_Fall"], errors="coerce")
    else:
        # Preserve columns for downstream display code (no functionality loss)
        data["P_Rise_log"] = np.nan
        data["P_Stay_log"] = np.nan
        data["P_Fall_log"] = np.nan

    # ------------------------------------------------------------------
    # Default active probabilities (Bin by default; Display toggle will switch)
    # ------------------------------------------------------------------
    data["P_Rise"] = data["P_Rise_bin"]
    data["P_Stay"] = data["P_Stay_bin"]
    data["P_Fall"] = data["P_Fall_bin"]

    pcols = ["P_Rise", "P_Stay", "P_Fall"]

    row_sum = data[pcols].sum(axis=1)
    has_probs = row_sum.notna() & (row_sum > 0)
    data.loc[has_probs, pcols] = data.loc[has_probs, pcols].div(row_sum[has_probs], axis=0)

    # Predicted class (argmax) ONLY where probabilities exist (for default engine)
    data["PredClass"] = np.where(
        has_probs,
        np.select(
            [
                (data["P_Rise"] >= data["P_Stay"]) & (data["P_Rise"] >= data["P_Fall"]),
                (data["P_Fall"] > data["P_Rise"]) & (data["P_Fall"] >= data["P_Stay"]),
            ],
            ["Rise", "Fall"],
            default="Stay",
        ),
        np.nan,
    )

    data["Correct"] = np.where(
        data["PredClass"].isna() | data["Y"].isna(),
        np.nan,
        (data["PredClass"] == data["Y"]).astype(float),
    )
    data["HitRate_50"] = data["Correct"].rolling(50, min_periods=10).mean()

    # Optional HMM (compute once)
    hmm_prob_df = None
    if show_hmm:
        hidden_states, prob_matrix = compute_hmm(data, n_states=3)
        data["HMM_State"] = hidden_states
        hmm_prob_df = pd.DataFrame(
            prob_matrix,
            columns=[f"Regime_{i}" for i in range(prob_matrix.shape[1])],
            index=data.index,
        )
        hmm_prob_df = _row_normalize(hmm_prob_df.shift(shift_days).bfill())

    # ------------------------------------------------------------------
    # Metrics (validation) for BOTH engines (Bin + Logistic)
    # ------------------------------------------------------------------
    def _eval_metrics(df_in: pd.DataFrame, cols: list[str]) -> dict:
        eval_df = df_in.dropna(subset=cols + ["Y"]).copy()
        y_to_idx = {"Rise": 0, "Stay": 1, "Fall": 2}
        if len(eval_df) == 0:
            return {"accuracy": np.nan, "logloss": np.nan}
        probs_np = eval_df[cols].to_numpy(dtype=float)
        y_idx = eval_df["Y"].map(y_to_idx).to_numpy(dtype=int)
        ll = _safe_log_loss(probs_np, y_idx)
        pred = np.select(
            [
                (eval_df[cols[0]] >= eval_df[cols[1]]) & (eval_df[cols[0]] >= eval_df[cols[2]]),
                (eval_df[cols[2]] > eval_df[cols[0]]) & (eval_df[cols[2]] >= eval_df[cols[1]]),
            ],
            ["Rise", "Fall"],
            default="Stay",
        )
        acc = float((pred == eval_df["Y"]).mean())
        return {"accuracy": acc, "logloss": ll}

    metrics_bin = _eval_metrics(data, ["P_Rise_bin", "P_Stay_bin", "P_Fall_bin"])
    metrics_log = _eval_metrics(data, ["P_Rise_log", "P_Stay_log", "P_Fall_log"])

    # ------------------------------------------------------------------
    # Live prediction for most recent day (BOTH engines)
    # ------------------------------------------------------------------
    latest_score = float(data["Score"].iloc[-1])

    # Bin live
    edges, tab, global_probs = fit_full_mapping_for_live_prediction(data, n_bins=n_bins)
    p_live_bin = predict_one_from_mapping(latest_score, edges, tab, global_probs)

    # Logistic live
    p_live_log = np.array([np.nan, np.nan, np.nan], dtype=float)
    pipe = fit_logistic_for_live_prediction(data, feature_cols=feature_cols, y_col="Y")
    if pipe is not None:
        last_row = data[feature_cols].iloc[-1]
        if last_row.notna().all():
            p_live_log = predict_one_logistic(pipe, last_row.to_numpy(dtype=float))

    # Store in session state for stable display
    st.session_state["v2_data"] = data
    st.session_state["v2_hmm_prob_df"] = hmm_prob_df
    st.session_state["v2_metrics"] = {
        "latest_score": latest_score,
        "bin": {**metrics_bin, "p_live": p_live_bin},
        "log": {**metrics_log, "p_live": p_live_log},
        "feature_cols": feature_cols,
    }

    st.success("Computed v2 results. Scroll down to charts and validation.")


# ------------------------
# Display
# ------------------------
if "v2_data" in st.session_state:
    data = st.session_state["v2_data"].copy()
    hmm_prob_df = st.session_state.get("v2_hmm_prob_df", None)
    metrics = st.session_state.get("v2_metrics", {})

    # ✅ Engine toggle (Bin vs Logistic) — remains (no functionality removed)
    engine_choice = st.radio(
        "Active engine for display",
        options=["bin", "log"],
        format_func=lambda x: "Bin (Score bins)" if x == "bin" else "Logistic (features)",
        horizontal=True,
        key="engine_choice",
    )

    # ✅ Apply chosen engine to active P_* columns
    if engine_choice == "log":
        data["P_Rise"] = data.get("P_Rise_log", np.nan)
        data["P_Stay"] = data.get("P_Stay_log", np.nan)
        data["P_Fall"] = data.get("P_Fall_log", np.nan)
    else:
        data["P_Rise"] = data.get("P_Rise_bin", np.nan)
        data["P_Stay"] = data.get("P_Stay_bin", np.nan)
        data["P_Fall"] = data.get("P_Fall_bin", np.nan)

    # Normalize chosen-engine rows (defensive)
    pcols = ["P_Rise", "P_Stay", "P_Fall"]
    row_sum = data[pcols].sum(axis=1)
    has_probs = row_sum.notna() & (row_sum > 0)
    data.loc[has_probs, pcols] = data.loc[has_probs, pcols].div(row_sum[has_probs], axis=0)

    # Recompute PredClass/Correct/HitRate_50 for chosen engine (so inspector/cal/backtest match)
    data["PredClass"] = np.where(
        has_probs,
        np.select(
            [
                (data["P_Rise"] >= data["P_Stay"]) & (data["P_Rise"] >= data["P_Fall"]),
                (data["P_Fall"] > data["P_Rise"]) & (data["P_Fall"] >= data["P_Stay"]),
            ],
            ["Rise", "Fall"],
            default="Stay",
        ),
        np.nan,
    )

    data["Correct"] = np.where(
        data["PredClass"].isna() | data["Y"].isna(),
        np.nan,
        (data["PredClass"] == data["Y"]).astype(float),
    )
    data["HitRate_50"] = data["Correct"].rolling(50, min_periods=10).mean()

    # Defensive numeric coercion (prevents Plotly odd behaviour on reruns)
    for c in ["Open", "High", "Low", "Close", "MA50", "MA200", "P_Rise", "P_Stay", "P_Fall", "Correct", "HitRate_50"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # ✅ Use engine-specific metrics
    engine_metrics = metrics.get(engine_choice, {})
    p_live = engine_metrics.get("p_live", np.array([np.nan, np.nan, np.nan]))
    latest_score = metrics.get("latest_score", np.nan)

    st.subheader("Latest horizon forecast (probabilities sum to 1)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horizon (days)", str(horizon))
    c2.metric("Latest Score", f"{latest_score:.3f}" if np.isfinite(latest_score) else "n/a")
    if np.all(np.isfinite(p_live)):
        c3.metric("P(Rise)", f"{p_live[0]:.3f}")
        c4.metric("P(Fall)", f"{p_live[2]:.3f}")
        st.write(f"**P(Stay)**: {p_live[1]:.3f}")
    else:
        c3.metric("P(Rise)", "n/a")
        c4.metric("P(Fall)", "n/a")
        st.write("**P(Stay)**: n/a")

    st.subheader("Validation summary (walk-forward)")
    acc = engine_metrics.get("accuracy", np.nan)
    ll = engine_metrics.get("logloss", np.nan)
    d1, d2 = st.columns(2)
    d1.metric("Walk-forward accuracy", f"{acc:.3f}" if np.isfinite(acc) else "n/a")
    d2.metric("Walk-forward log loss", f"{ll:.3f}" if np.isfinite(ll) else "n/a")

    non_nan_probs = int(data["P_Rise"].notna().sum()) if "P_Rise" in data.columns else 0
    if non_nan_probs == 0:
        st.warning("No walk-forward probability rows were produced yet. This can happen if the date range is too short for min_train=252.")
    elif non_nan_probs < 50:
        st.info(f"Only {non_nan_probs} probability rows were produced (walk-forward). Extend the date range for richer validation.")

    # ------------------------
    # Charts
    # ------------------------
    st.subheader("Charts")

    # Force Plotly to receive clean floats (prevents weird 0..N / diagonal scaling)
    def _float_list(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        return [None if pd.isna(v) else float(v) for v in s.to_numpy()]

    chart_mode = st.radio(
        "Price display",
        options=["Candlestick (OHLC)", "Close line (trend)"],
        index=0,
        horizontal=True,
        key="price_chart_mode",
    )

    first_prob_date = None
    if "P_Rise" in data.columns and data["P_Rise"].notna().any():
        first_prob_date = data.index[data["P_Rise"].notna()][0]

    x = data.index.to_pydatetime()

    # Chart 1: Price
    price_fig = go.Figure()
    if chart_mode.startswith("Candlestick"):
        price_fig.add_trace(
            go.Candlestick(
                x=x,
                open=_float_list(data["Open"]),
                high=_float_list(data["High"]),
                low=_float_list(data["Low"]),
                close=_float_list(data["Close"]),
                name="Price",
            )
        )
        price_fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        price_fig.add_trace(go.Scatter(x=x, y=_float_list(data["Close"]), name="Close", mode="lines"))

    price_fig.add_trace(go.Scatter(x=x, y=_float_list(data["MA50"]), name="MA50", mode="lines"))
    price_fig.add_trace(go.Scatter(x=x, y=_float_list(data["MA200"]), name="MA200", mode="lines"))

    if first_prob_date is not None:
        _add_wf_marker(price_fig, first_prob_date)

    price_fig.update_layout(title="Price + MA50/MA200", height=450, legend_title_text="Price")
    st.plotly_chart(price_fig, use_container_width=True)

    # Chart 2: Probabilities
    prob_fig = go.Figure()
    prob_fig.add_trace(go.Scatter(x=x, y=_float_list(data["P_Rise"]), name="P(Rise)", mode="lines"))
    prob_fig.add_trace(go.Scatter(x=x, y=_float_list(data["P_Stay"]), name="P(Stay)", mode="lines"))
    prob_fig.add_trace(go.Scatter(x=x, y=_float_list(data["P_Fall"]), name="P(Fall)", mode="lines"))

    if first_prob_date is not None:
        _add_wf_marker(prob_fig, first_prob_date)

    if y_zoom.startswith("0 to 1"):
        prob_fig.update_yaxes(range=[0, 1])
    else:
        prob_fig.update_yaxes(autorange=True)

    prob_fig.update_layout(
        title=f"Predicted probabilities for next {horizon} trading days (Rise/Stay/Fall)",
        height=350,
        legend_title_text="Probabilities",
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # Chart 3: Accuracy
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=x, y=_float_list(data["Correct"]), name="Correct (1/0)", mode="lines"))
    acc_fig.add_trace(go.Scatter(x=x, y=_float_list(data["HitRate_50"]), name="HitRate (50)", mode="lines"))

    if first_prob_date is not None:
        _add_wf_marker(acc_fig, first_prob_date)

    acc_fig.update_yaxes(range=[-0.05, 1.05])
    acc_fig.update_layout(
        title="Accuracy view (Correct=1/0) + Rolling Hit Rate (50)",
        height=300,
        legend_title_text="Accuracy",
    )
    st.plotly_chart(acc_fig, use_container_width=True)

    # Optional Chart 4: HMM
    if hmm_prob_df is not None:
        hmm_fig = go.Figure()
        hmm_prob_df = hmm_prob_df.copy()
        for c in hmm_prob_df.columns:
            hmm_prob_df[c] = pd.to_numeric(hmm_prob_df[c], errors="coerce")

        for c in hmm_prob_df.columns:
            hmm_fig.add_trace(go.Scatter(x=x, y=_float_list(hmm_prob_df[c]), name=c.replace("_", " "), mode="lines"))

        if first_prob_date is not None:
            _add_wf_marker(hmm_fig, first_prob_date)

        hmm_fig.update_yaxes(range=[0, 1])
        hmm_fig.update_layout(
            title=f"HMM regime probabilities (shifted by {shift_days} days)",
            height=300,
            legend_title_text="Regimes",
        )
        st.plotly_chart(hmm_fig, use_container_width=True)

    # ============================================================
    # A) Forecast Inspector (Point-in-time)
    # ============================================================
    st.subheader("Forecast inspector (pick a historical date)")

    inspect_df = data.dropna(subset=["P_Rise", "P_Stay", "P_Fall", "Y", "FwdRet_H"]).copy()

    if inspect_df.empty:
        st.warning("No rows available for inspection (need probabilities + realized outcome). Try a longer date range.")
    else:
        inspect_dates = inspect_df.index
        chosen_date = st.selectbox(
            "Select a date to inspect (model belief at that time)",
            options=list(inspect_dates),
            index=len(inspect_dates) - 1,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="inspect_date",
        )

        row = inspect_df.loc[chosen_date]

        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Date", chosen_date.strftime("%Y-%m-%d"))
        i2.metric("Predicted", str(row.get("PredClass", "n/a")))
        i3.metric("Actual (Y)", str(row.get("Y", "n/a")))
        i4.metric("Forward return (H)", f"{float(row['FwdRet_H']):.3%}")
        i5.metric("Correct", "✅" if float(row.get("Correct", 0)) == 1 else "❌")

        p1, p2, p3 = st.columns(3)
        p1.metric("P(Rise)", f"{float(row['P_Rise']):.3f}")
        p2.metric("P(Stay)", f"{float(row['P_Stay']):.3f}")
        p3.metric("P(Fall)", f"{float(row['P_Fall']):.3f}")

        st.caption("Local context around the selected date (± 40 trading days)")
        window = 40
        pos = data.index.get_loc(chosen_date)
        left = max(0, pos - window)
        right = min(len(data.index) - 1, pos + window)
        ctx = data.iloc[left: right + 1].copy()

        ctx_price_fig = go.Figure()
        ctx_price_fig.add_trace(go.Scatter(x=ctx.index, y=_float_list(ctx["Close"]), name="Close", mode="lines"))
        ctx_price_fig.add_trace(go.Scatter(x=ctx.index, y=_float_list(ctx["MA50"]), name="MA50", mode="lines"))
        ctx_price_fig.add_trace(go.Scatter(x=ctx.index, y=_float_list(ctx["MA200"]), name="MA200", mode="lines"))
        _add_wf_marker(ctx_price_fig, chosen_date, label="Selected date")
        ctx_price_fig.update_layout(height=300, title="Local price context")
        st.plotly_chart(ctx_price_fig, use_container_width=True)

        ctx_prob_fig = go.Figure()
        ctx_prob_fig.add_trace(go.Scatter(x=ctx.index, y=_float_list(ctx["P_Rise"]), name="P(Rise)", mode="lines"))
        ctx_prob_fig.add_trace(go.Scatter(x=ctx.index, y=_float_list(ctx["P_Stay"]), name="P(Stay)", mode="lines"))
        ctx_prob_fig.add_trace(go.Scatter(x=ctx.index, y=_float_list(ctx["P_Fall"]), name="P(Fall)", mode="lines"))
        _add_wf_marker(ctx_prob_fig, chosen_date, label="Selected date")
        ctx_prob_fig.update_yaxes(range=[0, 1])
        ctx_prob_fig.update_layout(height=300, title="Local probability context")
        st.plotly_chart(ctx_prob_fig, use_container_width=True)

    # ============================================================
    # B) Calibration plot (reliability)
    # ============================================================
    st.subheader("Calibration (do predicted probabilities match reality?)")

    cal_df = data.dropna(subset=["P_Rise", "P_Stay", "P_Fall", "Y", "PredClass"]).copy()
    if cal_df.empty:
        st.warning("No rows available for calibration (need probabilities + realized outcome).")
    else:
        def pred_prob_of_row(r):
            pc = r["PredClass"]
            if pc == "Rise":
                return r["P_Rise"]
            if pc == "Stay":
                return r["P_Stay"]
            if pc == "Fall":
                return r["P_Fall"]
            return np.nan

        cal_df["P_pred"] = cal_df.apply(pred_prob_of_row, axis=1)
        cal_df = cal_df.dropna(subset=["P_pred"])
        cal_df["IsCorrect"] = (cal_df["PredClass"] == cal_df["Y"]).astype(int)

        n_bins_cal = st.slider(
            "Calibration bins",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key="cal_bins",
        )

        cal_df["bin"] = pd.cut(cal_df["P_pred"], bins=n_bins_cal, include_lowest=True)

        grp = cal_df.groupby("bin", observed=False).agg(
            mean_pred=("P_pred", "mean"),
            frac_correct=("IsCorrect", "mean"),
            count=("IsCorrect", "size"),
        ).dropna()

        cal_fig = go.Figure()

        # Perfect reference line (thin, dashed)
        cal_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Perfect (y=x)",
                mode="lines",
                line=dict(dash="dash", width=1),
            )
        )

        # Observed calibration (markers only, so it cannot hide)
        cal_fig.add_trace(
            go.Scatter(
                x=grp["mean_pred"],
                y=grp["frac_correct"],
                name="Observed",
                mode="markers",
                marker=dict(size=10),
            )
        )

        cal_fig.update_xaxes(range=[0, 1], title="Mean predicted probability (of predicted class)")
        cal_fig.update_yaxes(range=[0, 1], title="Observed frequency (prediction was correct)")
        cal_fig.update_layout(height=350, title="Calibration plot (reliability diagram)")
        st.plotly_chart(cal_fig, use_container_width=True)

        st.caption("Bin counts (support behind each point):")
        st.dataframe(grp[["count"]].reset_index(drop=False), use_container_width=True)
    # ============================================================
    # D) Economic backtest view (simple, transparent)
    # ============================================================
    st.subheader("Economic backtest (simple rules)")

    bt_df = data.dropna(subset=["P_Rise", "P_Stay", "P_Fall", "FwdRet_H"]).copy()
    if bt_df.empty:
        st.warning("No rows available for backtest (need probabilities + forward return).")
    else:
        cA, cB, cC = st.columns(3)
        threshold = cA.slider("Trade threshold (min winning-class probability)", 0.33, 0.80, 0.45, 0.01, key="bt_thresh")
        long_only = cB.checkbox("Long-only (ignore short signals)", value=True, key="bt_long_only")
        non_overlap = cC.checkbox("Non-overlapping trades (every H days)", value=True, key="bt_non_overlap")

        probs = bt_df[["P_Rise", "P_Stay", "P_Fall"]].to_numpy(dtype=float)
        max_idx = np.argmax(probs, axis=1)
        max_p = np.max(probs, axis=1)

        signal = np.where(max_p >= threshold, np.select([max_idx == 0, max_idx == 2], [1, -1], default=0), 0)
        if long_only:
            signal = np.where(signal == -1, 0, signal)

        bt_df["Signal"] = signal
        bt_df["TradeRet_H"] = bt_df["Signal"] * bt_df["FwdRet_H"]

        if non_overlap:
            trade_mask = np.zeros(len(bt_df), dtype=bool)
            i = 0
            while i < len(bt_df):
                if bt_df["Signal"].iloc[i] != 0:
                    trade_mask[i] = True
                    i += int(horizon)
                else:
                    i += 1
            bt_df["TradeActive"] = trade_mask
            bt_trades = bt_df[bt_df["TradeActive"]].copy()
        else:
            bt_trades = bt_df[bt_df["Signal"] != 0].copy()

        if bt_trades.empty:
            st.info("No trades triggered under current threshold/settings. Lower the threshold to see activity.")
        else:
            bt_trades["Equity"] = (1.0 + bt_trades["TradeRet_H"]).cumprod()

            bh = data.loc[bt_trades.index.min(): bt_trades.index.max()].copy()
            bh["BH"] = (bh["Close"] / bh["Close"].iloc[0]).astype(float)

            # Force numeric to avoid Plotly treating series as categorical
            bt_trades["Equity"] = pd.to_numeric(bt_trades["Equity"], errors="coerce")
            bh["BH"] = pd.to_numeric(bh["BH"], errors="coerce")

            bt_fig = go.Figure()
            bt_fig.add_trace(
                go.Scatter(
                    x=bt_trades.index,
                    y=_float_list(bt_trades["Equity"]),
                    name="Strategy equity",
                    mode="lines",
                )
            )
            bt_fig.add_trace(
                go.Scatter(
                    x=bh.index,
                    y=_float_list(bh["BH"]),
                    name="Buy & hold (scaled)",
                    mode="lines",
                )
            )
            bt_fig.update_layout(height=350, title="Backtest: Strategy vs Buy & Hold (scaled to 1.0)")
            st.plotly_chart(bt_fig, use_container_width=True)

            total_return = float(bt_trades["Equity"].iloc[-1] - 1.0)
            n_trades = int(len(bt_trades))
            win_rate = float((bt_trades["TradeRet_H"] > 0).mean())
            avg_trade = float(bt_trades["TradeRet_H"].mean())

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Trades", str(n_trades))
            s2.metric("Win rate", f"{win_rate:.3f}")
            s3.metric("Avg trade (H)", f"{avg_trade:.3%}")
            s4.metric("Total return", f"{total_return:.3%}")

            st.caption("Most recent trades:")
            show_bt_cols = ["Close", "Signal", "P_Rise", "P_Stay", "P_Fall", "FwdRet_H", "TradeRet_H"]
            show_bt_cols = [c for c in show_bt_cols if c in bt_trades.columns]
            st.dataframe(bt_trades[show_bt_cols].tail(10).round(6), use_container_width=True)

    # ------------------------
    # Table
    # ------------------------
    st.subheader("Recent rows (for inspection)")
    show_cols = [
        "Close", "Volume",
        "Score",
        "sigma_t", "delta_t_H",
        "FwdRet_H", "Y",
        "P_Rise", "P_Stay", "P_Fall",
        "PredClass", "Correct",
    ]
    existing = [c for c in show_cols if c in data.columns]
    st.dataframe(data[existing].tail(15).round(6), use_container_width=True)

    if st.checkbox("Show class balance (labeled data only)", value=False):
        labeled = data.dropna(subset=["Y"])
        st.write(labeled["Y"].value_counts(normalize=True).round(4).to_dict())

    if st.checkbox("Show probability row-sums (tail)", value=False):
        sums = (data["P_Rise"] + data["P_Stay"] + data["P_Fall"]).tail(10)
        st.write(sums.round(6).tolist())