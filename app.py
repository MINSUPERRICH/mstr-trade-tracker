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
#  MATH ENGINE (Black-Scholes)
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
#  APP LAYOUT
# =========================================================
st.title("🚀 MSTR Option Command Center")

# --- SIDEBAR ---
st.sidebar.header("📝 Trade Settings")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.markdown("---")
symbol = st.sidebar.text_input("Symbol", value="MSTR").upper()
current_stock_price = st.sidebar.number_input("Current Stock Price ($)", value=158.00, step=0.50)
strike_price = st.sidebar.number_input("Strike Price ($)", value=157.50, step=0.50)
expiration_date = st.sidebar.date_input("Expiration Date", value=date(2026, 1, 9))
purchase_date = st.sidebar.date_input("Purchase Date", value=date(2024, 12, 24))
entry_price = st.sidebar.number_input("Entry Price", value=8.55, step=0.10)
contracts = st.sidebar.number_input("Contracts", value=1, step=1)
implied_volatility = st.sidebar.slider("Implied Volatility (IV %)", 10, 200, 95) / 100.0
risk_free_rate = 0.045

# --- TABS ---
tab_math, tab_ai = st.tabs(["🔮 Profit Simulator (Math)", "📸 Chart Analyst (AI)"])

# =========================================================
#  TAB 1: MATH SIMULATOR + Q&A
# =========================================================
with tab_math:
    st.subheader(f"📊 {symbol} Price Simulator")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_date = st.slider("📅 Future Date", min_value=purchase_date, max_value=expiration_date, value=date.today() + timedelta(days=5), format="MMM DD")
    with col_sim2:
        sim_price = st.slider("💲 Future Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0)

    # Math Calculations
    days_to_expiry_sim = (expiration_date - sim_date).days
    time_to_expiry_years = max(days_to_expiry_sim / 365.0, 0.0001)
    projected_option_price = black_scholes(sim_price, strike_price, time_to_expiry_years, risk_free_rate, implied_volatility)
    net_profit = (projected_option_price * 100 * contracts) - (entry_price * 100 * contracts)
    
    # Display Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Date", sim_date.strftime('%b %d'))
    m2.metric("Stock Price", f"${sim_price:.2f}")
    m3.metric("Option Price", f"${projected_option_price:.2f}")
    m4.metric("Net Profit", f"${net_profit:,.2f}", delta_color="normal" if net_profit >= 0 else "inverse")

    # Heatmap
    st.markdown("---")
    prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
    future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)]
    heatmap_data = []
    for d in future_dates:
        t_years = max((expiration_date - d).days / 365.0, 0.0001)
        for p in prices:
            opt = black_scholes(p, strike_price, t_years, risk_free_rate, implied_volatility)
            pl = (opt - entry_price) * 100 * contracts
            heatmap_data.append({"Date": d.strftime('%Y-%m-%d'), "Stock Price": round(p, 2), "Profit": round(pl, 2)})
            
    c = alt.Chart(pd.DataFrame(heatmap_data)).mark_rect().encode(
        x='Date:O', y='Stock Price:O', color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)), tooltip=['Date', 'Stock Price', 'Profit']
    ).properties(height=350)
    st.altair_chart(c, use_container_width=True)

    # --- Q&A FOR MATH TAB ---
    st.markdown("### 💬 Ask about this Scenario")
    math_question = st.text_input("Ask a question about these numbers (e.g., 'Why am I losing money?')", key="math_q")
    
    if st.button("Ask Gemini (Simulator)"):
        if not api_key:
            st.error("Missing API Key")
        else:
            with st.spinner("Thinking..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    context = f"""
                    You are a financial trading expert. The user is simulating a {symbol} Call Option.
                    Current Inputs:
                    - Buy Price: ${entry_price}
                    - Simulation Date: {sim_date}
                    - Simulated Stock Price: ${sim_price}
                    - Resulting Profit/Loss: ${net_profit}
                    - Days Passed: {(sim_date - purchase_date).days}
                    
                    User Question: {math_question}
                    """
                    response = model.generate_content(context)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")

# =========================================================
#  TAB 2: AI ANALYST (MULTI-FILE SUPPORT)
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Chart Analysis")
    st.markdown("Upload **one or multiple** charts. Example: Upload an **Option Chain** and a **Stock Chart** to compare them.")
    
    # 🆕 CHANGED: accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload Screenshots...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        # Load all images
        images = [Image.open(f) for f in uploaded_files]
        
        # Display images side-by-side (max 3 per row to look good)
        cols = st.columns(len(images))
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Chart {i+1}", use_container_width=True)
        
        # Store in session state for follow-up questions
        st.session_state["last_images"] = images

        # Initial Analysis Button
        if st.button("✨ Analyze All Files"):
            if not api_key:
                st.error("Missing API Key")
            else:
                with st.spinner("Gemini is comparing your files..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        prompt = f"""
                        You are an expert options trader specializing in {symbol}.
                        I have provided {len(images)} images. They may be Technical Charts, Option Chains (Strike/Expiration), or Heatmaps.

                        Please perform a **Comparative Analysis**:
                        1. **Synthesize:** Look at all images together. Do they tell the same story, or do they contradict?
                        2. **Sentiment:** Is the data across these files Bullish or Bearish?
                        3. **Strategy:** I am holding a Long Call (Strike ${strike_price}, Exp {expiration_date}). 
                        
                        Based on ALL provided evidence, give a recommendation.
                        """
                        
                        # Send text + all images to Gemini
                        content = [prompt] + images
                        response = model.generate_content(content)
                        
                        st.markdown("### 🧠 Gemini's Verdict:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")

        # --- Q&A FOR AI TAB ---
        st.markdown("---")
        st.markdown("### 💬 Ask about these Charts")
        chart_question = st.text_input("Ask a follow-up question (e.g., 'What about the volume in the second image?')", key="chart_q")
        
        if st.button("Ask Gemini (Chart)"):
            if "last_images" not in st.session_state:
                st.warning("Please upload charts first.")
            elif not api_key:
                st.error("Missing API Key")
            else:
                with st.spinner("Reviewing evidence..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        # We send the question + ALL stored images
                        content = [chart_question] + st.session_state["last_images"]
                        response = model.generate_content(content)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
