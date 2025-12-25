import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from scipy.stats import norm
from PIL import Image
import io
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
#  TAB 1: MATH SIMULATOR + DOWNLOADS
# =========================================================
with tab_math:
    st.subheader(f"📊 {symbol} Price Simulator")
    
    # FORCE RECALCULATION BUTTON
    if st.button("🔄 Force Recalculate (Update Dates)"):
        st.rerun()

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        default_sim_date = max(purchase_date, date.today())
        sim_date = st.slider("📅 Future Date", min_value=purchase_date, max_value=expiration_date, value=default_sim_date, format="MMM DD")
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

    # Heatmap Calculation
    st.markdown("---")
    prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
    future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)]
    heatmap_data = []
    
    for d in future_dates:
        t_years = max((expiration_date - d).days / 365.0, 0.0001)
        for p in prices:
            opt = black_scholes(p, strike_price, t_years, risk_free_rate, implied_volatility)
            pl = (opt - entry_price) * 100 * contracts
            heatmap_data.append({
                "Date": d.strftime('%Y-%m-%d'), 
                "Stock Price": round(p, 2), 
                "Option Price": round(opt, 2),
                "Profit": round(pl, 2)
            })
            
    df_heatmap = pd.DataFrame(heatmap_data)

    # DOWNLOAD BUTTON FOR SIMULATOR CSV
    csv = df_heatmap.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Simulator Data (CSV)",
        data=csv,
        file_name=f"{symbol}_simulation_{date.today()}.csv",
        mime="text/csv",
    )

    # Chart Display
    c = alt.Chart(df_heatmap).mark_rect().encode(
        x='Date:O', y='Stock Price:O', color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)), tooltip=['Date', 'Stock Price', 'Profit']
    ).properties(height=350)
    st.altair_chart(c, use_container_width=True)

    # --- Q&A FOR MATH TAB ---
    st.markdown("### 💬 Ask about this Scenario")
    
    # Initialize session state for Q&A Text
    if "math_ai_response" not in st.session_state:
        st.session_state["math_ai_response"] = ""

    math_question = st.text_input("Ask a question about these numbers...", key="math_q")
    
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
                    
                    CRITICAL TRADE DETAILS:
                    - Strike Price: ${strike_price}
                    - Expiration Date: {expiration_date}
                    - Purchase Price: ${entry_price}
                    - Implied Volatility: {implied_volatility * 100}%
                    
                    SIMULATION SCENARIO:
                    - Simulated Date: {sim_date}
                    - Simulated Stock Price: ${sim_price}
                    - Simulated Option Price: ${projected_option_price}
                    - Simulated Net Profit: ${net_profit}
                    
                    User Question: {math_question}
                    
                    Please explain the math or strategy based on these exact numbers.
                    """
                    
                    response = model.generate_content(context)
                    st.session_state["math_ai_response"] = response.text
                    
                except Exception as e:
                    st.error(f"Error: {e}")

    # Display Response & Download Button
    if st.session_state["math_ai_response"]:
        st.info(st.session_state["math_ai_response"])
        
        # 🆕 NEW DOWNLOAD BUTTON FOR Q&A
        st.download_button(
            label="📥 Download Gemini Explanation (.txt)",
            data=st.session_state["math_ai_response"],
            file_name=f"Gemini_Scenario_Explanation_{date.today()}.txt",
            mime="text/plain"
        )

# =========================================================
#  TAB 2: AI ANALYST + DOWNLOADS
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Chart Analysis")
    uploaded_files = st.file_uploader("Upload Screenshots...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    # Initialize session state for AI response if not exists
    if "ai_analysis_text" not in st.session_state:
        st.session_state["ai_analysis_text"] = ""

    if uploaded_files:
        images = [Image.open(f) for f in uploaded_files]
        st.session_state["last_images"] = images
        
        cols = st.columns(len(images))
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Chart {i+1}", use_container_width=True)

        if st.button("✨ Analyze All Files"):
            if not api_key:
                st.error("Missing API Key")
            else:
                with st.spinner("Gemini is analyzing..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        prompt = f"""
                        You are an expert options trader specializing in {symbol}.
                        Analyze these {len(images)} images (Technical Charts/Option Chains).
                        1. Compare the data.
                        2. Is the sentiment Bullish or Bearish?
                        3. I hold a Long Call (Strike ${strike_price}, Exp {expiration_date}). Recommendation?
                        """
                        
                        content = [prompt] + images
                        response = model.generate_content(content)
                        
                        # Store response in session state
                        st.session_state["ai_analysis_text"] = response.text
                        st.markdown("### 🧠 Gemini's Verdict:")
                        st.write(response.text)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

        # DOWNLOAD BUTTON FOR AI ANALYSIS
        if st.session_state["ai_analysis_text"]:
            st.download_button(
                label="📥 Download AI Report (.txt)",
                data=st.session_state["ai_analysis_text"],
                file_name=f"{symbol}_AI_Analysis_{date.today()}.txt",
                mime="text/plain"
            )

        # --- Q&A FOR AI TAB ---
        st.markdown("---")
        chart_question = st.text_input("Ask a follow-up question...", key="chart_q")
        
        if st.button("Ask Gemini (Chart)"):
            if "last_images" not in st.session_state:
                st.warning("Please upload charts first.")
            elif not api_key:
                st.error("Missing API Key")
            else:
                with st.spinner("Reviewing..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        content = [chart_question] + st.session_state["last_images"]
                        response = model.generate_content(content)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
