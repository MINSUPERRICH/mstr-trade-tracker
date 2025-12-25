import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MSTR Trade Command",
    page_icon="🔒",
    layout="centered"
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

    # 1. If we are already authenticated, return True
    if st.session_state.get("password_correct", False):
        return True

    # 2. Show the Input Box
    st.title("🔒 Locked")
    st.text_input(
        "Please enter the password to access MSTR Command:", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    # 3. Handle Errors
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

# --- STOP HERE IF NOT LOGGED IN ---
if not check_password():
    st.stop()  # The app halts here. Nothing below runs.

# =========================================================
#  ✅ AUTHORIZED ZONE: EVERYTHING BELOW IS YOUR APP
# =========================================================

# --- HEADER ---
st.title("🚀 MSTR Option Command Center")
st.markdown("### Strategy: Long Call | Jan 9, 2026 | Strike $157.5")
st.write("---")

# --- SIDEBAR: SECURE API LOADING ---
st.sidebar.header("🔑 AI Access")

# Try to get key from Secrets first
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ API Key Loaded Securely")
else:
    # Fallback to manual entry
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if not api_key:
        st.sidebar.warning("⚠️ Key required for AI features")

st.sidebar.header("⚙️ Scenario Simulator")
current_price = st.sidebar.number_input("Current MSTR Stock Price ($)", value=157.88, step=0.50)
days_passed = st.sidebar.slider("Days Since Purchase (Dec 24)", 0, 16, 0)

# --- CONSTANTS ---
ENTRY_PRICE = 8.55
STRIKE = 157.50
CONTRACTS = 1
INITIAL_EXTRINSIC = 8.17 
THETA_DAILY = 0.50 

# --- CALCULATIONS ---
intrinsic_value = max(0, current_price - STRIKE)
extrinsic_value = max(0, INITIAL_EXTRINSIC - (days_passed * THETA_DAILY))
estimated_option_price = intrinsic_value + extrinsic_value
total_cost = ENTRY_PRICE * 100 * CONTRACTS
current_value_total = estimated_option_price * 100 * CONTRACTS
profit_loss = current_value_total - total_cost
roi = (profit_loss / total_cost) * 100

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Est. Option Price", f"${estimated_option_price:.2f}", delta=f"{estimated_option_price - ENTRY_PRICE:.2f}")
with col2:
    st.metric("Your Profit/Loss", f"${profit_loss:.0f}", delta=f"{roi:.1f}%")
with col3:
    st.metric("Ice Cube Melted", f"-${days_passed * THETA_DAILY * 100:.0f}", delta_color="inverse")

if days_passed > 5 and profit_loss < 0:
    st.warning(f"⚠️ **Theta Warning:** You have held this for {days_passed} days. Consider exiting.")

# --- TABS SECTION ---
st.write("---")
tab1, tab2, tab3, tab4 = st.tabs(["📸 AI Analyst", "🚀 Exit Targets", "🛡️ Emergency Stop", "📉 Fidelity Guide"])

# --- TAB 1: AI ANALYST ---
with tab1:
    st.subheader("🤖 AI Chart Analysis")
    st.write("Upload your option chart. Gemini will analyze volume, sentiment, and risk for you.")
    
    uploaded_file = st.file_uploader("Choose a screenshot...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Chart', use_container_width=True)
        
        if st.button("✨ Analyze with Gemini"):
            if not api_key:
                st.error("Please provide an API Key in secrets or sidebar!")
            else:
                with st.spinner("Gemini is studying the chart..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        prompt = """
                        You are an expert options trader. Analyze this chart image. 
                        1. Identify if this is a Bullish or Bearish setup.
                        2. Look at Volume bars: Is the crowd buying Calls or Puts?
                        3. Calculate the Breakeven point if visible.
                        4. Give a one-sentence recommendation for a Long Call holder.
                        """
                        
                        response = model.generate_content([prompt, image])
                        st.markdown("### 🧠 Gemini's Verdict:")
                        st.write(response.text)
                        
                    except Exception as e:
                        st.error(f"Error connecting to Gemini: {e}")

# --- TAB 2: EXIT TARGETS ---
with tab2:
    st.success("### 🎯 Profit Target: $15.00")
    st.write(f"**Trigger:** Set a Limit Sell Order (GTC) at **$15.00**.")
    st.write("If filled, you make **~$645 Profit (+75%)**.")

# --- TAB 3: STOP LOSS ---
with tab3:
    st.error("### 🛑 Stop Loss: $6.00")
    st.write("**Trigger:** If option price drops to **$6.00**, SELL IMMEDIATELY.")
    st.write("This limits your loss to **~$255**.")

# --- TAB 4: FIDELITY GUIDE ---
with tab4:
    st.info("### 🏦 How to set 'Robot' on Fidelity")
    st.markdown("""
    1. Go to **Accounts & Trade** > **Trade** on the website.
    2. Change "Trade Type" to **Conditional**.
    3. Select **One Cancels the Other (OCO)**.
    4. **Order 1 (Win):** Sell Limit @ **$15.00** (GTC).
    5. **Order 2 (Safe):** Sell Stop Loss @ **$6.00** (GTC).
    """)

# --- MELTING ICE CUBE CHART ---
st.write("---")
st.subheader("🧊 Visualizing the 'Melting Ice Cube'")
days = list(range(0, 17))
decay_values = [ENTRY_PRICE - (d * 0.50) for d in days] 
chart_data = pd.DataFrame({'Days From Now': days, 'Option Value (If Stock Flat)': decay_values})
c = alt.Chart(chart_data).mark_line(color='#FF4B4B', point=True).encode(
    x='Days From Now', y=alt.Y('Option Value (If Stock Flat)', scale=alt.Scale(domain=[0, 10])),
    tooltip=['Days From Now', 'Option Value (If Stock Flat)']
).interactive()

st.altair_chart(c, use_container_width=True)
