import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from scipy.stats import norm
from PIL import Image
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MSTR Command Center",
    page_icon="🚀",
    layout="wide"
)

# --- SECURITY SYSTEM (THE GATEKEEPER) ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password in plain text
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Locked")
    st.text_input(
        "Please enter the password to access MSTR Command:", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

# --- STOP HERE IF NOT LOGGED IN ---
if not check_password():
    st.stop()

# =========================================================
#  MATH ENGINE: BLACK-SCHOLES FORMULA
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

# =========================================================
#  APP LAYOUT & INPUTS
# =========================================================
st.title("🚀 MSTR Option Command Center")

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.header("📝 Trade Settings")

# API Key Handling
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("Contract Details")
symbol = st.sidebar.text_input("Stock Symbol", value="MSTR").upper()
current_stock_price = st.sidebar.number_input("Current Stock Price ($)", value=158.00, step=0.50)
strike_price = st.sidebar.number_input("Strike Price ($)", value=157.50, step=0.50)
expiration_date = st.sidebar.date_input("Expiration Date", value=date(2026, 1, 9))
purchase_date = st.sidebar.date_input("Purchase Date", value=date(2024, 12, 24))
entry_price = st.sidebar.number_input("Entry Price (Premium Paid)", value=8.55, step=0.10)
contracts = st.sidebar.number_input("Number of Contracts", value=1, step=1)
implied_volatility = st.sidebar.slider("Implied Volatility (IV %)", 10, 200, 95) / 100.0
risk_free_rate = 0.045

# --- TABS FOR DIFFERENT BRAINS ---
tab_math, tab_ai = st.tabs(["🔮 Profit Simulator (Math)", "📸 Chart Analyst (AI)"])

# =========================================================
#  TAB 1: THE SIMULATOR (MATH)
# =========================================================
with tab_math:
    st.subheader(f"📊 {symbol} Price Simulator")
    st.write("Adjust the sliders to see what your option is worth in different future scenarios.")

    # Simulation Sliders
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_date = st.slider(
            "📅 Future Date", 
            min_value=purchase_date, 
            max_value=expiration_date, 
            value=date.today() + timedelta(days=5),
            format="MMM DD, YYYY"
        )
    with col_sim2:
        sim_price = st.slider(
            "💲 Future Stock Price", 
            min_value=float(current_stock_price * 0.5), 
            max_value=float(current_stock_price * 2.0), 
            value=float(current_stock_price),
            step=1.0
        )

    # Calculations
    days_to_expiry_sim = (expiration_date - sim_date).days
    time_to_expiry_years = max(days_to_expiry_sim / 365.0, 0.0001)
    projected_option_price = black_scholes(sim_price, strike_price, time_to_expiry_years, risk_free_rate, implied_volatility)
    
    total_cost = entry_price * 100 * contracts
    exit_value = projected_option_price * 100 * contracts
    net_profit = exit_value - total_cost
    roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

    # Display Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target Date", sim_date.strftime('%b %d'), f"{(sim_date - date.today()).days} days away")
    m2.metric("Target Stock", f"${sim_price:.2f}")
    m3.metric("Est. Option Value", f"${projected_option_price:.2f}", delta=f"{projected_option_price - entry_price:.2f}")
    m4.metric("Net Profit", f"${net_profit:,.2f}", delta=f"{roi:.1f}%", delta_color="normal" if net_profit >= 0 else "inverse")

    # Heatmap
    st.write("---")
    st.write("### 🗺️ Profit Heatmap (Price vs Date)")
    prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
    future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)]
    heatmap_data = []

    for d in future_dates:
        t_years = max((expiration_date - d).days / 365.0, 0.0001)
        for p in prices:
            opt_price = black_scholes(p, strike_price, t_years, risk_free_rate, implied_volatility)
            pl = (opt_price - entry_price) * 100 * contracts
            heatmap_data.append({"Date": d.strftime('%Y-%m-%d'), "Stock Price": round(p, 2), "Profit": round(pl, 2)})

    df_heatmap = pd.DataFrame(heatmap_data)
    heatmap = alt.Chart(df_heatmap).mark_rect().encode(
        x='Date:O', y='Stock Price:O',
        color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
        tooltip=['Date', 'Stock Price', 'Profit']
    ).properties(height=400)
    st.altair_chart(heatmap, use_container_width=True)

# =========================================================
#  TAB 2: THE ANALYST (AI)
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Chart Analysis")
    st.markdown("""
    **How to use this:**
    1. Take a screenshot of the Option Chain (showing Volume/Open Interest) or the Stock Chart.
    2. Upload it below.
    3. Gemini will analyze the "Crowd Psychology" for you.
    """)
    
    uploaded_file = st.file_uploader("Upload screenshot...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Chart', use_container_width=True)
        
        if st.button("✨ Analyze with Gemini"):
            if not api_key:
                st.error("Please provide an API Key in secrets or sidebar!")
            else:
                with st.spinner("Gemini is analyzing market sentiment..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        prompt = f"""
                        You are an expert options trader specializing in {symbol}. 
                        Analyze this image. It is likely an option chain or a technical chart.
                        
                        1. **Sentiment Check:** Based on the visual data (volume bars, trend lines, or option open interest), is the crowd Bullish or Bearish?
                        2. **Key Levels:** Identify any support/resistance or "walls" of Open Interest.
                        3. **Strategy Fit:** I am holding a Long Call (Strike ${strike_price}, Exp {expiration_date}). Does this image support holding or selling?
                        4. **Verdict:** Give a clear 1-sentence recommendation.
                        """
                        
                        response = model.generate_content([prompt, image])
                        st.markdown("### 🧠 Gemini's Verdict:")
                        st.write(response.text)
                        
                    except Exception as e:
                        st.error(f"Error connecting to Gemini: {e}")
