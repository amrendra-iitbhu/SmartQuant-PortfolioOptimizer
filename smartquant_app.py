import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import cvxpy as cp
from datetime import datetime, timedelta

# === Streamlit Setup ===
st.set_page_config(
    page_title="SmartQuant Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Professional Theme (CSS) ===
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
    }

    .app-header {
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid #262730;
        margin-bottom: 1.5rem;
    }

    .app-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        color: #fafafa;
        letter-spacing: -0.02em;
    }

    .app-header p {
        font-size: 0.95rem;
        color: #9ca3af;
        margin: 0.25rem 0 0 0;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: #161a23;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #3b3f4a;
    }

    .stAlert {
        border-radius: 8px;
    }

    hr {
        border-color: #262730;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #262730;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>SmartQuant Pro</h1>
        <p>Machine learning-driven portfolio optimization and risk analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# === Sidebar Inputs ===
st.sidebar.markdown("### Configuration")

tickers = st.sidebar.multiselect(
    "Assets",
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'NVDA', 'META', 'XOM'],
    default=['AAPL', 'MSFT', 'GOOGL'],
)

default_start = pd.to_datetime("2023-01-01")
default_end = pd.to_datetime("today")

st.sidebar.markdown("---")
real_time = st.sidebar.checkbox("Enable real-time updates (1-min interval)", value=False)

if real_time:
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    interval = "1m"
else:
    start = st.sidebar.date_input("Start date", default_start)
    end = st.sidebar.date_input("End date", default_end)
    interval = "1d"

model_choice = st.sidebar.radio("ML model", ['Lasso', 'Random Forest'])

if real_time:
    if st.sidebar.button("Refresh now"):
        st.rerun()

if not tickers:
    st.info("Select at least one asset from the sidebar to begin.")
    st.stop()


# === Load Data ===
@st.cache_data(ttl=60)
def load_data(tickers, start=None, end=None, interval="1d"):

    if interval == "1m":
        df = yf.download(
            tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
    else:
        df = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )

    if df.empty:
        return pd.DataFrame()

    prices = pd.DataFrame()

    if len(tickers) == 1:
        if "Close" in df.columns:
            prices[tickers[0]] = df["Close"]
        elif "Adj Close" in df.columns:
            prices[tickers[0]] = df["Adj Close"]
    else:
        for ticker in tickers:
            try:
                if "Close" in df[ticker]:
                    prices[ticker] = df[ticker]["Close"]
                elif "Adj Close" in df[ticker]:
                    prices[ticker] = df[ticker]["Adj Close"]
            except Exception as e:
                st.warning(f"{ticker}: {e}")

    return prices.ffill().dropna(how="all")


data = load_data(tickers, start, end, interval)

if data.empty:
    st.error("No data fetched. Try selecting different tickers or disabling real-time mode.")
    st.stop()


# === Feature Engineering ===
def compute_features(prices):
    df = pd.DataFrame(index=prices.index)
    df['return_1d'] = prices.pct_change()
    df['volatility_5d'] = prices.pct_change().rolling(5).std()
    df['momentum_5d'] = prices / prices.shift(5) - 1
    df['sma_diff'] = prices.rolling(5).mean() - prices.rolling(10).mean()
    df['target'] = prices.pct_change().shift(-1)
    return df.dropna()


# === Train ML Models ===
pred_all = []

for ticker in tickers:
    df = compute_features(data[ticker])
    features = ['return_1d', 'volatility_5d', 'momentum_5d', 'sma_diff']
    X, y = df[features], df['target']

    if model_choice == 'Lasso':
        model = Lasso(alpha=0.001)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    try:
        model.fit(X, y)
        pred = model.predict(X[-100:])
        pred_all.append(np.mean(pred))
    except Exception as e:
        st.warning(f"Model training failed for {ticker}: {e}")
        pred_all.append(0.0)

# === Optimization ===
returns = np.array(pred_all)
cov = data[tickers].pct_change().dropna().cov().values
w = cp.Variable(len(tickers))
objective = cp.Minimize(cp.quad_form(w, cov))
constraints = [cp.sum(w) == 1, w >= 0, returns @ w >= np.mean(returns)]
prob = cp.Problem(objective, constraints)
prob.solve()

if prob.status != 'optimal':
    st.error(f"Optimization failed using {model_choice}. Try fewer assets or a different model.")
    st.stop()

weights = w.value
weights = np.clip(weights, 0, None)
weights = weights / weights.sum()

# === Portfolio Returns ===
daily_ret = data[tickers].pct_change().dropna()
port_ret = daily_ret @ weights
cum_ret = (1 + port_ret).cumprod()

# === Charts ===
st.markdown('<div class="section-label">Portfolio Overview</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Allocation**")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    fig1.patch.set_alpha(0.0)
    ax1.set_facecolor("none")
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(tickers)))
    wedges, texts, autotexts = ax1.pie(
        weights,
        labels=tickers,
        autopct='%1.1f%%',
        colors=colors,
        textprops={'color': '#e5e7eb', 'fontsize': 10},
    )
    st.pyplot(fig1, transparent=True)

with col2:
    st.markdown("**Cumulative Return**")
    st.line_chart(cum_ret)

# === Risk Metrics ===
st.markdown('<div class="section-label">Risk Analysis</div>', unsafe_allow_html=True)

VaR_95 = -np.percentile(port_ret, 5)
tail_losses = port_ret[port_ret <= -VaR_95]
CVaR_95 = -tail_losses.mean() if len(tail_losses) > 0 else float('nan')
drawdown = cum_ret / cum_ret.cummax() - 1

col1, col2, col3 = st.columns(3)
col1.metric("Value at Risk (95%)", f"{VaR_95:.2%}")
col2.metric("Conditional VaR (95%)", f"{CVaR_95:.2%}" if not np.isnan(CVaR_95) else "N/A")
col3.metric("Max Drawdown", f"{drawdown.min():.2%}")

st.markdown("**Rolling 20-Day Volatility**")
st.line_chart(port_ret.rolling(20).std())