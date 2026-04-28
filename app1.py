# app.py
# ============================================================
# 🚀 Hammad Quant Pro v2
# Professional Stock Forecasting Dashboard
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hammad Quant Pro",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.main{
background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
color:white;
}
section[data-testid="stSidebar"]{
background:#0f172a;
}
.stButton>button{
background:linear-gradient(90deg,#06b6d4,#2563eb);
color:white;
border:none;
border-radius:10px;
font-weight:bold;
}
.stDownloadButton>button{
background:linear-gradient(90deg,#10b981,#059669);
color:white;
border:none;
border-radius:10px;
}
.footer{
position:fixed;
bottom:0;
left:0;
width:100%;
padding:10px;
background:#0f172a;
text-align:center;
color:white;
z-index:999;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.title("📈 Hammad Quant Pro")
st.markdown("""
### AI Stock Forecasting Dashboard  
Forecast prices using **Prophet** + **Random Forest**
""")

# ============================================================
# DATA LOADER
# ============================================================
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.columns = [str(c).strip() for c in df.columns]

    return df

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.selectbox(
    "Choose Stock",
    [
        "AAPL","MSFT","GOOG","META","TSLA",
        "NVDA","AMD","AMZN","NFLX","INTC"
    ]
)

start_date = st.sidebar.date_input(
    "Start Date",
    date.today() - timedelta(days=365*3)
)

end_date = st.sidebar.date_input(
    "End Date",
    date.today()
)

model_name = st.sidebar.selectbox(
    "Choose Model",
    ["Prophet", "Random Forest"]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    7, 365, 30
)

run = st.sidebar.button("🚀 Run Analysis")

# ============================================================
# MAIN
# ============================================================
if run:

    with st.spinner("Downloading market data..."):
        data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found.")
        st.stop()

    # ========================================================
    # METRICS
    # ========================================================
    c1, c2, c3, c4 = st.columns(4)

    close = round(float(data["Close"].iloc[-1]),2)
    high = round(float(data["High"].max()),2)
    low = round(float(data["Low"].min()),2)
    volume = int(data["Volume"].iloc[-1])

    c1.metric("Last Close", close)
    c2.metric("Highest", high)
    c3.metric("Lowest", low)
    c4.metric("Volume", f"{volume:,}")

    st.markdown("---")

    # ========================================================
    # CANDLESTICK
    # ========================================================
    st.subheader("📊 Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )])

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # DATA TABLE
    # ========================================================
    with st.expander("📄 Show Raw Data"):
        st.dataframe(data.tail(50), use_container_width=True)

    # ========================================================
    # PROPHET
    # ========================================================
    if model_name == "Prophet":

        st.subheader("🔮 Prophet Forecast")

        dfp = data[["Date", "Close"]].rename(
            columns={"Date":"ds","Close":"y"}
        )

        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(dfp)

        future = model.make_future_dataframe(
            periods=forecast_days
        )

        forecast = model.predict(future)

        fig2 = px.line(
            forecast,
            x="ds",
            y="yhat",
            title="Forecast Trend"
        )

        st.plotly_chart(fig2, use_container_width=True)

        out = forecast[
            ["ds","yhat","yhat_lower","yhat_upper"]
        ].tail(forecast_days)

        st.dataframe(out, use_container_width=True)

        csv = out.to_csv(index=False).encode()

        st.download_button(
            "⬇️ Download Forecast CSV",
            csv,
            "forecast.csv",
            "text/csv"
        )

    # ========================================================
    # RANDOM FOREST
    # ========================================================
    else:

        st.subheader("🌲 Random Forest Forecast")

        df = data[["Date","Close"]].copy()
        df["Day"] = np.arange(len(df))

        X = df[["Day"]]
        y = df["Close"]

        split = int(len(df)*0.8)

        X_train = X[:split]
        X_test = X[split:]

        y_train = y[:split]
        y_test = y[split:]

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        m1,m2,m3 = st.columns(3)

        m1.metric("RMSE", round(rmse,2))
        m2.metric("MAE", round(mae,2))
        m3.metric("R²", round(r2,3))

        # Plot
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Actual"
        ))

        fig3.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=pred,
            name="Predicted"
        ))

        fig3.update_layout(height=550)

        st.plotly_chart(fig3, use_container_width=True)

        # Future forecast
        future_x = np.arange(
            len(df),
            len(df)+forecast_days
        ).reshape(-1,1)

        future_pred = model.predict(future_x)

        future_dates = pd.date_range(
            start=df["Date"].iloc[-1] + timedelta(days=1),
            periods=forecast_days
        )

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_pred
        })

        st.subheader("📅 Future Forecast")

        st.dataframe(future_df, use_container_width=True)

        csv = future_df.to_csv(index=False).encode()

        st.download_button(
            "⬇️ Download Forecast CSV",
            csv,
            "future_forecast.csv",
            "text/csv"
        )

    # ========================================================
    # SIMPLE SIGNAL
    # ========================================================
    st.subheader("📌 AI Signal")

    ma20 = data["Close"].rolling(20).mean().iloc[-1]
    last = data["Close"].iloc[-1]

    if last > ma20:
        st.success("🟢 BUY Signal (Price above 20MA)")
    else:
        st.error("🔴 SELL Signal (Price below 20MA)")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
Made with ❤️ by <b>Hammad Zahid</b> | AI • Finance • Quant Research
</div>
""", unsafe_allow_html=True)