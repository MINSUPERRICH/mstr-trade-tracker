import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from scipy.stats import norm
from PIL import Image
import io
import google.generativeai as genai
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MSTR Command Center",
    page_icon="🚀",
    layout="wide"
)

# --- SECURITY SYSTEM ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Locked")
    st.text_input("Enter Password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

# =========================================================
#  MATH ENGINE (Black-Scholes & DSS Bressert)
# =========================================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_dss_bressert(ticker, period=10, ema_period=9):
    """
    Calculates the Bressert Double Smoothed Stochastic.
    Formula: EMA( Stochastic( EMA( Stochastic(Price) ) ) )
    """
    try:
        # Fetch data (need enough history for smoothing)
        df = yf.download(ticker, period="3mo", progress=False)
        if len(df) < period + ema_period:
            return None, None

        # 1. High/Low/Close
        high = df['High']
        low = df['Low']
        close = df['Close']

        # 2. First Stochastic Calculation (Raw)
        # (Close - LowestLow) / (HighestHigh - LowestLow)
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_raw = (close - lowest_low) / (highest_high - lowest_low) * 100

        # 3. First Smoothing (EMA of Raw Stoch)
        xPreCalc = stoch_raw.ewm(span=ema_period, adjust=False).mean()

        # 4. Second Stochastic Calculation (Stochastic of the Smoothed Line)
        # Treat xPreCalc as the "Price" -> Find its range
        lowest_smooth = xPreCalc.rolling(window=period).min()
        highest_smooth = xPreCalc.rolling(window=period).max()
        
        # Avoid division by zero
        denominator = highest_smooth - lowest_smooth
        denominator = denominator.replace(0, 1) 
        
        stoch_smooth = (xPreCalc - lowest_smooth) / denominator * 100

        # 5. Final Smoothing (EMA of the Second Stoch) -> The DSS
        dss = stoch_smooth.ewm(span=ema_period, adjust=False).mean()

        current_val = dss.iloc[-1]
        
        # Determine Status
        if isinstance(current_val, pd.Series):
             current_val = current_val.item()

        return round(current_val, 2), df['Close'].iloc[-1].item()
        
    except Exception as e:
        return None, None

# =========================================================
#  APP LAYOUT
# =========================================================
st.title("🚀 MSTR Option Command Center")

# --- SIDEBAR (Market Conditions) ---
st.sidebar.header("🌍 Market Conditions")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.markdown("---")
symbol = st.sidebar.text_input("Symbol", value="MSTR").upper()
current_stock_price = st.sidebar.number_input("Current Stock Price ($)", value=158.00, step=0.50)
implied_volatility = st.sidebar.slider("Implied Volatility (IV %)", 10, 200, 95) / 100.0
risk_free_rate = 0.045

st.sidebar.info("💡 **Tip:** Go to 'Market Dashboard' to scan multiple stocks.")

# --- TABS ---
tab_math, tab_dashboard, tab_ai = st.tabs(["⚔️ Strategy Battle", "📊 Market Dashboard", "📸 Chart Analyst"])

# =========================================================
#  TAB 1: SCENARIO BATTLE (UNCHANGED)
# =========================================================
with tab_math:
    st.subheader(f"⚖️ Compare Strategies")
    if st.button("🔄 Reset Scenarios"):
        st.rerun()

    col_a, col_b = st.columns(2)

    # SCENARIO A
    with col_a:
        st.info("### 🔵 Strategy A")
        strike_a = st.number_input("Strike Price ($)", value=157.50, step=0.50, key="strike_a")
        exp_date_a = st.date_input("Expiration Date", value=date(2026, 1, 9), key="exp_a")
        entry_price_a = st.number_input("Entry Price (Paid)", value=8.55, step=0.10, key="entry_a")
        contracts_a = st.number_input("Contracts", value=1, step=1, key="count_a")
        st.markdown("---")
        sim_date_a = st.slider("📅 Sell Date", min_value=date.today(), max_value=exp_date_a, value=date.today() + timedelta(days=5), key="date_a", format="MMM DD")
        sim_price_a = st.slider("💲 Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="price_a")
        
        days_a = (exp_date_a - sim_date_a).days
        years_a = max(days_a / 365.0, 0.0001)
        opt_price_a = black_scholes(sim_price_a, strike_a, years_a, risk_free_rate, implied_volatility)
        profit_a = (opt_price_a * 100 * contracts_a) - (entry_price_a * 100 * contracts_a)
        st.metric("Net Profit (A)", f"${profit_a:,.2f}", delta_color="normal" if profit_a >= 0 else "inverse")

    # SCENARIO B
    with col_b:
        st.warning("### 🟠 Strategy B")
        strike_b = st.number_input("Strike Price ($)", value=157.50, step=0.50, key="strike_b")
        exp_date_b = st.date_input("Expiration Date", value=date(2026, 1, 9), key="exp_b")
        entry_price_b = st.number_input("Entry Price (Paid)", value=8.55, step=0.10, key="entry_b")
        contracts_b = st.number_input("Contracts", value=1, step=1, key="count_b")
        st.markdown("---")
        sim_date_b = st.slider("📅 Sell Date", min_value=date.today(), max_value=exp_date_b, value=date.today() + timedelta(days=5), key="date_b", format="MMM DD")
        sim_price_b = st.slider("💲 Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="price_b")
        
        days_b = (exp_date_b - sim_date_b).days
        years_b = max(days_b / 365.0, 0.0001)
        opt_price_b = black_scholes(sim_price_b, strike_b, years_b, risk_free_rate, implied_volatility)
        profit_b = (opt_price_b * 100 * contracts_b) - (entry_price_b * 100 * contracts_b)
        st.metric("Net Profit (B)", f"${profit_b:,.2f}", delta_color="normal" if profit_b >= 0 else "inverse")

    # HEATMAP & DECAY (Simplified for brevity, insert full heatmap/decay code if desired or keep simple comparison)
    st.write("---")
    diff = profit_a - profit_b
    if diff > 0:
        st.success(f"🏆 Strategy A Wins by ${diff:,.2f}")
    else:
        st.warning(f"🏆 Strategy B Wins by ${abs(diff):,.2f}")

# =========================================================
#  TAB 2: MARKET DASHBOARD (DSS BRESSERT) [NEW]
# =========================================================
with tab_dashboard:
    st.subheader("📊 DSS Bressert Scanner")
    st.markdown("""
    Monitor multiple stocks using the **Double Smoothed Stochastic (DSS)**.
    * **Buy Zone (Green):** DSS < 20 (Oversold)
    * **Sell Zone (Red):** DSS > 80 (Overbought)
    """)

    # Input for tickers
    default_tickers = "MSTR, BTC-USD, COIN, NVDA, IBIT, MSTU"
    ticker_input = st.text_input("Enter Tickers (comma separated)", value=default_tickers)
    
    if st.button("🔎 Scan Market"):
        tickers = [t.strip().upper() for t in ticker_input.split(",")]
        results = []

        progress_bar = st.progress(0)
        
        for i, tick in enumerate(tickers):
            dss_val, price = calculate_dss_bressert(tick)
            
            if dss_val is not None:
                status = "Neutral"
                color = "gray"
                if dss_val <= 20:
                    status = "🟢 OVERSOLD (Buy Watch)"
                    color = "green"
                elif dss_val >= 80:
                    status = "🔴 OVERBOUGHT (Sell Watch)"
                    color = "red"
                
                results.append({
                    "Ticker": tick,
                    "Price": f"${price:,.2f}",
                    "DSS Value": dss_val,
                    "Status": status
                })
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()

        if results:
            df_res = pd.DataFrame(results)
            
            # Styling function for the dataframe
            def color_status(val):
                if "OVERSOLD" in val:
                    return 'background-color: #d4edda; color: green; font-weight: bold'
                elif "OVERBOUGHT" in val:
                    return 'background-color: #f8d7da; color: red; font-weight: bold'
                return ''

            st.dataframe(
                df_res.style.applymap(color_status, subset=['Status']),
                use_container_width=True,
                height=400
            )
        else:
            st.error("No data found. Check ticker symbols.")

# =========================================================
#  TAB 3: AI ANALYST (UNCHANGED)
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Chart Analysis")
    uploaded_files = st.file_uploader("Upload Screenshots...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    if "ai_analysis_text" not in st.session_state:
        st.session_state["ai_analysis_text"] = ""

    if uploaded_files:
        images = [Image.open(f) for f in uploaded_files]
        st.session_state["last_images"] = images
        
        if st.button("✨ Analyze All Files"):
            if not api_key:
                st.error("Missing API Key")
            else:
                with st.spinner("Gemini is analyzing..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        prompt = f"Analyze these {len(images)} images for {symbol}. Bullish or Bearish?"
                        response = model.generate_content([prompt] + images)
                        st.session_state["ai_analysis_text"] = response.text
                        st.markdown("### 🧠 Gemini's Verdict:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.session_state["ai_analysis_text"]:
             st.download_button("📥 Download Report", st.session_state["ai_analysis_text"], "AI_Report.txt")
