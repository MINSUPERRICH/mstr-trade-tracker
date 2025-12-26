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

# --- SIDEBAR (Global Settings) ---
st.sidebar.header("📝 Global Trade Settings")
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
tab_math, tab_ai = st.tabs(["⚔️ Scenario Battle (Compare A vs B)", "📸 Chart Analyst (AI)"])

# =========================================================
#  TAB 1: SCENARIO BATTLE (A vs B)
# =========================================================
with tab_math:
    st.subheader(f"⚖️ Compare Two Outcomes")
    
    if st.button("🔄 Reset Scenarios"):
        st.rerun()

    # --- LAYOUT: TWO COLUMNS ---
    col_a, col_b = st.columns(2)

    # ==========================
    # 🔵 SCENARIO A (LEFT)
    # ==========================
    with col_a:
        st.info("### 🔵 Scenario A (Plan A)")
        
        # Sliders for A
        sim_date_a = st.slider("📅 Date (Scenario A)", min_value=purchase_date, max_value=expiration_date, value=date.today() + timedelta(days=5), key="date_a", format="MMM DD")
        sim_price_a = st.slider("💲 Stock Price (Scenario A)", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="price_a")
        
        # Math for A
        days_a = (expiration_date - sim_date_a).days
        years_a = max(days_a / 365.0, 0.0001)
        opt_price_a = black_scholes(sim_price_a, strike_price, years_a, risk_free_rate, implied_volatility)
        profit_a = (opt_price_a * 100 * contracts) - (entry_price * 100 * contracts)
        
        # Display A
        st.markdown(f"**Option Value:** ${opt_price_a:.2f}")
        st.metric("Net Profit (A)", f"${profit_a:,.2f}", delta_color="normal" if profit_a >= 0 else "inverse")

    # ==========================
    # 🟠 SCENARIO B (RIGHT)
    # ==========================
    with col_b:
        st.warning("### 🟠 Scenario B (Plan B)")
        
        # Sliders for B
        sim_date_b = st.slider("📅 Date (Scenario B)", min_value=purchase_date, max_value=expiration_date, value=date.today() + timedelta(days=20), key="date_b", format="MMM DD")
        sim_price_b = st.slider("💲 Stock Price (Scenario B)", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price * 1.1), step=1.0, key="price_b")
        
        # Math for B
        days_b = (expiration_date - sim_date_b).days
        years_b = max(days_b / 365.0, 0.0001)
        opt_price_b = black_scholes(sim_price_b, strike_price, years_b, risk_free_rate, implied_volatility)
        profit_b = (opt_price_b * 100 * contracts) - (entry_price * 100 * contracts)
        
        # Display B
        st.markdown(f"**Option Value:** ${opt_price_b:.2f}")
        st.metric("Net Profit (B)", f"${profit_b:,.2f}", delta_color="normal" if profit_b >= 0 else "inverse")

    # ==========================
    # 🏆 THE COMPARISON RESULT
    # ==========================
    st.write("---")
    diff = profit_a - profit_b
    
    if diff > 0:
        st.success(f"🏆 **Scenario A Wins!** It makes **${diff:,.2f}** more than Scenario B.")
    elif diff < 0:
        st.warning(f"🏆 **Scenario B Wins!** It makes **${abs(diff):,.2f}** more than Scenario A.")
    else:
        st.info("🤝 Both Scenarios result in the exact same profit.")

    # --- DOWNLOAD DATA ---
    st.markdown("### 📥 Export Data")
    
    # Create simple dataframe for export
    data = {
        "Metric": ["Date", "Stock Price", "Option Price", "Total Profit"],
        "Scenario A": [sim_date_a, sim_price_a, round(opt_price_a, 2), round(profit_a, 2)],
        "Scenario B": [sim_date_b, sim_price_b, round(opt_price_b, 2), round(profit_b, 2)],
        "Difference (A - B)": [f"{(sim_date_a - sim_date_b).days} days", round(sim_price_a - sim_price_b, 2), round(opt_price_a - opt_price_b, 2), round(profit_a - profit_b, 2)]
    }
    df_compare = pd.DataFrame(data)
    
    csv = df_compare.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Comparison CSV",
        data=csv,
        file_name=f"Comparison_{date.today()}.csv",
        mime="text/csv",
    )

    # --- Q&A FOR COMPARISON ---
    st.markdown("---")
    st.markdown("### 💬 Ask about this Comparison")
    
    if "compare_ai_response" not in st.session_state:
        st.session_state["compare_ai_response"] = ""

    comp_question = st.text_input("Ask a question (e.g., 'Is waiting for Scenario B worth the risk?')", key="comp_q")
    
    if st.button("Ask Gemini (Comparison)"):
        if not api_key:
            st.error("Missing API Key")
        else:
            with st.spinner("Analyzing both scenarios..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    context = f"""
                    You are a trading expert. User is comparing two scenarios for a {symbol} Call Option.
                    
                    GLOBAL SETTINGS:
                    - Strike: ${strike_price}, Expiration: {expiration_date}, Buy Price: ${entry_price}
                    
                    SCENARIO A (BLUE):
                    - Date: {sim_date_a}
                    - Stock Price: ${sim_price_a}
                    - Net Profit: ${profit_a}
                    
                    SCENARIO B (ORANGE):
                    - Date: {sim_date_b}
                    - Stock Price: ${sim_price_b}
                    - Net Profit: ${profit_b}
                    
                    User Question: {comp_question}
                    
                    Compare the risks (Time Decay in B vs A) and rewards. Which path seems safer?
                    """
                    
                    response = model.generate_content(context)
                    st.session_state["compare_ai_response"] = response.text
                    
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state["compare_ai_response"]:
        st.info(st.session_state["compare_ai_response"])
        st.download_button(
            label="📥 Download Explanation",
            data=st.session_state["compare_ai_response"],
            file_name="Gemini_Comparison.txt"
        )

# =========================================================
#  TAB 2: AI ANALYST (UNCHANGED)
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Chart Analysis")
    uploaded_files = st.file_uploader("Upload Screenshots...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
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
                        Analyze these {len(images)} images.
                        1. Compare the data.
                        2. Is the sentiment Bullish or Bearish?
                        3. I hold a Long Call (Strike ${strike_price}, Exp {expiration_date}). Recommendation?
                        """
                        
                        content = [prompt] + images
                        response = model.generate_content(content)
                        st.session_state["ai_analysis_text"] = response.text
                        st.markdown("### 🧠 Gemini's Verdict:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state["ai_analysis_text"]:
            st.download_button(
                label="📥 Download AI Report",
                data=st.session_state["ai_analysis_text"],
                file_name=f"{symbol}_AI_Analysis.txt"
            )

        st.markdown("---")
        if "chart_q_response" not in st.session_state:
            st.session_state["chart_q_response"] = ""

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
                        st.session_state["chart_q_response"] = response.text
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state["chart_q_response"]:
            st.info(st.session_state["chart_q_response"])
            st.download_button(
                label="📥 Download Q&A Response",
                data=st.session_state["chart_q_response"],
                file_name=f"Gemini_Chart_QA.txt"
            )
