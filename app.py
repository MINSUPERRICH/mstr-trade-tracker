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

def calculate_dss_data(ticker, period=10, ema_period=9):
    """
    Calculates DSS and returns the full DataFrame for charting.
    """
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        if len(df) < period + ema_period:
            return None

        # Handle Multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        high = df['High']
        low = df['Low']
        close = df['Close']

        # DSS Calculation Logic
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_raw = (close - lowest_low) / (highest_high - lowest_low) * 100

        xPreCalc = stoch_raw.ewm(span=ema_period, adjust=False).mean()

        lowest_smooth = xPreCalc.rolling(window=period).min()
        highest_smooth = xPreCalc.rolling(window=period).max()
        
        denominator = highest_smooth - lowest_smooth
        denominator = denominator.replace(0, 1) 
        
        stoch_smooth = (xPreCalc - lowest_smooth) / denominator * 100
        dss = stoch_smooth.ewm(span=ema_period, adjust=False).mean()

        # Add DSS to dataframe
        df['DSS'] = dss
        
        # Return only relevant columns, drop NaN
        return df[['Close', 'DSS']].dropna().reset_index()
        
    except Exception as e:
        return None

# =========================================================
#  APP LAYOUT
# =========================================================
st.title("🚀 MSTR Option Command Center")

# --- SIDEBAR ---
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

# --- TABS ---
tab_math, tab_dashboard, tab_ai = st.tabs(["⚔️ Strategy Battle", "📊 Market Dashboard", "📸 Chart Analyst"])

# =========================================================
#  TAB 1: STRATEGY BATTLE (UNCHANGED)
# =========================================================
with tab_math:
    st.subheader(f"⚖️ Compare Strategies")
    if st.button("🔄 Reset Scenarios"):
        st.rerun()

    col_a, col_b = st.columns(2)

    # --- SCENARIO A ---
    with col_a:
        st.info("### 🔵 Strategy A")
        st.markdown("**1. Contract Details**")
        strike_a = st.number_input("Strike ($)", value=157.50, step=0.50, key="str_a")
        exp_date_a = st.date_input("Expiration", value=date(2026, 1, 9), key="exp_a")
        entry_price_a = st.number_input("Entry Price", value=8.55, step=0.10, key="ent_a")
        contracts_a = st.number_input("Contracts", value=1, step=1, key="cnt_a")
        
        st.markdown("**2. Future Scenario**")
        sim_date_a = st.slider("Sell Date", min_value=date.today(), max_value=exp_date_a, value=date.today() + timedelta(days=5), key="d_a", format="MMM DD")
        sim_price_a = st.slider("Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="p_a")
        
        days_a = (exp_date_a - sim_date_a).days
        years_a = max(days_a / 365.0, 0.0001)
        opt_price_a = black_scholes(sim_price_a, strike_a, years_a, risk_free_rate, implied_volatility)
        profit_a = (opt_price_a * 100 * contracts_a) - (entry_price_a * 100 * contracts_a)
        
        st.metric("Est. Value", f"${opt_price_a:.2f}")
        st.metric("Net Profit (A)", f"${profit_a:,.2f}", delta_color="normal" if profit_a >= 0 else "inverse")

    # --- SCENARIO B ---
    with col_b:
        st.warning("### 🟠 Strategy B")
        st.markdown("**1. Contract Details**")
        strike_b = st.number_input("Strike ($)", value=157.50, step=0.50, key="str_b")
        exp_date_b = st.date_input("Expiration", value=date(2026, 1, 9), key="exp_b")
        entry_price_b = st.number_input("Entry Price", value=8.55, step=0.10, key="ent_b")
        contracts_b = st.number_input("Contracts", value=1, step=1, key="cnt_b")
        
        st.markdown("**2. Future Scenario**")
        sim_date_b = st.slider("Sell Date", min_value=date.today(), max_value=exp_date_b, value=date.today() + timedelta(days=5), key="d_b", format="MMM DD")
        sim_price_b = st.slider("Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="p_b")
        
        days_b = (exp_date_b - sim_date_b).days
        years_b = max(days_b / 365.0, 0.0001)
        opt_price_b = black_scholes(sim_price_b, strike_b, years_b, risk_free_rate, implied_volatility)
        profit_b = (opt_price_b * 100 * contracts_b) - (entry_price_b * 100 * contracts_b)
        
        st.metric("Est. Value", f"${opt_price_b:.2f}")
        st.metric("Net Profit (B)", f"${profit_b:,.2f}", delta_color="normal" if profit_b >= 0 else "inverse")

    # --- WINNER BANNER ---
    st.write("---")
    diff = profit_a - profit_b
    if diff > 0:
        st.success(f"🏆 **Strategy A Wins!** (+${diff:,.2f})")
    elif diff < 0:
        st.warning(f"🏆 **Strategy B Wins!** (+${abs(diff):,.2f})")
    else:
        st.info("🤝 Draw")

    # --- DOWNLOAD COMPARISON CSV ---
    data_comp = {
        "Metric": ["Strike", "Exp", "Sim Date", "Stock Price", "Profit"],
        "Strategy A": [strike_a, exp_date_a, sim_date_a, sim_price_a, round(profit_a, 2)],
        "Strategy B": [strike_b, exp_date_b, sim_date_b, sim_price_b, round(profit_b, 2)]
    }
    st.download_button("📥 Download Comparison CSV", pd.DataFrame(data_comp).to_csv(index=False).encode('utf-8'), "Comparison.csv", "text/csv")

    # --- HEATMAP SECTION ---
    st.markdown("---")
    st.subheader("🗺️ Profit Heatmap")
    map_choice = st.radio("Show Map for:", ["Strategy A 🔵", "Strategy B 🟠"], horizontal=True)
    
    if map_choice == "Strategy A 🔵":
        h_strike, h_exp, h_entry, h_contracts = strike_a, exp_date_a, entry_price_a, contracts_a
    else:
        h_strike, h_exp, h_entry, h_contracts = strike_b, exp_date_b, entry_price_b, contracts_b

    prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
    future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)]
    heatmap_data = []
    
    for d in future_dates:
        t_years = max((h_exp - d).days / 365.0, 0.0001)
        for p in prices:
            opt = black_scholes(p, h_strike, t_years, risk_free_rate, implied_volatility)
            pl = (opt - h_entry) * 100 * h_contracts
            heatmap_data.append({"Date": d.strftime('%Y-%m-%d'), "Stock Price": round(p, 2), "Profit": round(pl, 2)})
            
    df_heatmap = pd.DataFrame(heatmap_data)
    
    c = alt.Chart(df_heatmap).mark_rect().encode(
        x='Date:O', y='Stock Price:O', 
        color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
        tooltip=['Date', 'Stock Price', 'Profit']
    ).properties(height=350)
    st.altair_chart(c, use_container_width=True)
    
    st.download_button("📥 Download Heatmap CSV", df_heatmap.to_csv(index=False).encode('utf-8'), "Heatmap.csv", "text/csv")

    # --- TIME DECAY SECTION ---
    st.markdown("---")
    st.subheader("📉 Time Decay Comparison")
    decay_data = []
    for i in range(120): # Next 120 days
        d = date.today() + timedelta(days=i)
        
        # A
        if d < exp_date_a:
            ta = max((exp_date_a - d).days / 365.0, 0.0001)
            va = black_scholes(sim_price_a, strike_a, ta, risk_free_rate, implied_volatility)
            decay_data.append({"Date": d, "Value": va, "Strategy": "Strategy A 🔵"})
        
        # B
        if d < exp_date_b:
            tb = max((exp_date_b - d).days / 365.0, 0.0001)
            vb = black_scholes(sim_price_b, strike_b, tb, risk_free_rate, implied_volatility)
            decay_data.append({"Date": d, "Value": vb, "Strategy": "Strategy B 🟠"})

    decay_chart = alt.Chart(pd.DataFrame(decay_data)).mark_line(strokeWidth=3).encode(
        x='Date:T', y='Value:Q', color='Strategy', tooltip=['Date', 'Value']
    ).properties(height=350).interactive()
    st.altair_chart(decay_chart, use_container_width=True)

    # --- Q&A SECTION (TAB 1) ---
    st.markdown("---")
    st.markdown("### 💬 Ask about this Comparison")
    
    if "compare_ai_response" not in st.session_state: st.session_state["compare_ai_response"] = ""
    comp_q = st.text_input("Ask a question about these scenarios...", key="comp_q")
    
    if st.button("Ask Gemini (Comparison)"):
        if not api_key: st.error("Missing API Key")
        else:
            with st.spinner("Analyzing..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    ctx = f"""Comparing options. 
                    A: Strike {strike_a}, Exp {exp_date_a}, Profit {profit_a}.
                    B: Strike {strike_b}, Exp {exp_date_b}, Profit {profit_b}.
                    User Q: {comp_q}"""
                    response = model.generate_content(ctx)
                    st.session_state["compare_ai_response"] = response.text
                except Exception as e: st.error(f"Error: {e}")

    if st.session_state["compare_ai_response"]:
        st.info(st.session_state["compare_ai_response"])
        st.download_button("📥 Download Explanation", st.session_state["compare_ai_response"], "Gemini_Battle.txt")

# =========================================================
#  TAB 2: MARKET DASHBOARD (UPDATED WITH GRAPH)
# =========================================================
with tab_dashboard:
    st.subheader("📊 DSS Bressert Scanner & Chart")
    st.markdown("Scanner for **Double Smoothed Stochastic**. Green < 20 (Buy), Red > 80 (Sell).")
    
    default_tickers = "MSTR, BTC-USD, COIN, NVDA, IBIT, MSTU"
    ticker_input = st.text_input("Enter Tickers (comma separated)", value=default_tickers)
    
    # Session state to hold scan results so they don't disappear when we click other buttons
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = []
    
    if st.button("🔎 Scan Market"):
        tickers = [t.strip().upper() for t in ticker_input.split(",")]
        results = []
        progress = st.progress(0)
        
        for i, tick in enumerate(tickers):
            # We use calculate_dss_data just to get the last value for the table
            df_hist = calculate_dss_data(tick)
            
            if df_hist is not None and not df_hist.empty:
                dss_val = df_hist['DSS'].iloc[-1]
                price = df_hist['Close'].iloc[-1]
                
                status = "Neutral"
                if dss_val <= 20: status = "🟢 OVERSOLD (Buy Watch)"
                elif dss_val >= 80: status = "🔴 OVERBOUGHT (Sell Watch)"
                
                results.append({"Ticker": tick, "Price": f"${price:,.2f}", "DSS": round(dss_val, 2), "Status": status})
            
            progress.progress((i + 1) / len(tickers))
        
        progress.empty()
        st.session_state["scan_results"] = results

    # Display Table if results exist
    if st.session_state["scan_results"]:
        df_res = pd.DataFrame(st.session_state["scan_results"])
        
        def color_status(val):
            if "OVERSOLD" in val: return 'background-color: #d4edda; color: green; font-weight: bold'
            elif "OVERBOUGHT" in val: return 'background-color: #f8d7da; color: red; font-weight: bold'
            return ''
            
        st.dataframe(df_res.style.applymap(color_status, subset=['Status']), use_container_width=True)
        
        # --- NEW GRAPH SECTION ---
        st.write("---")
        st.subheader("📉 DSS Indicator Chart")
        
        # Dropdown to pick a ticker found in the scan
        found_tickers = [r["Ticker"] for r in st.session_state["scan_results"]]
        selected_ticker = st.selectbox("Select Ticker to Chart:", found_tickers)
        
        if selected_ticker:
            with st.spinner(f"Loading chart for {selected_ticker}..."):
                df_chart = calculate_dss_data(selected_ticker)
                
                if df_chart is not None:
                    # Base Chart
                    base = alt.Chart(df_chart).encode(x='Date:T')
                    
                    # DSS Line
                    line = base.mark_line(color='#1f77b4', strokeWidth=2).encode(
                        y=alt.Y('DSS', scale=alt.Scale(domain=[0, 100])),
                        tooltip=['Date', 'DSS', 'Close']
                    )
                    
                    # Overbought Line (80)
                    line_80 = alt.Chart(pd.DataFrame({'y': [80]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
                    
                    # Oversold Line (20)
                    line_20 = alt.Chart(pd.DataFrame({'y': [20]})).mark_rule(color='green', strokeDash=[5, 5]).encode(y='y')
                    
                    # Combine
                    chart = (line + line_80 + line_20).properties(height=350).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)

# =========================================================
#  TAB 3: AI ANALYST (UNCHANGED)
# =========================================================
with tab_ai:
    st.subheader("🤖 AI Chart Analysis")
    uploaded_files = st.file_uploader("Upload Screenshots...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    if "ai_analysis_text" not in st.session_state: st.session_state["ai_analysis_text"] = ""
    if "chart_q_response" not in st.session_state: st.session_state["chart_q_response"] = ""

    if uploaded_files:
        images = [Image.open(f) for f in uploaded_files]
        st.session_state["last_images"] = images
        
        cols = st.columns(len(images))
        for i, img in enumerate(images):
            with cols[i]: st.image(img, caption=f"Chart {i+1}", use_container_width=True)

        if st.button("✨ Analyze All Files"):
            if not api_key: st.error("Missing API Key")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        prompt = f"Analyze these {len(images)} images for {symbol}. Bullish or Bearish?"
                        response = model.generate_content([prompt] + images)
                        st.session_state["ai_analysis_text"] = response.text
                    except Exception as e: st.error(f"Error: {e}")
        
        if st.session_state["ai_analysis_text"]:
            st.markdown("### 🧠 Gemini's Verdict:")
            st.write(st.session_state["ai_analysis_text"])
            st.download_button("📥 Download Report", st.session_state["ai_analysis_text"], "AI_Report.txt")

        st.markdown("---")
        chart_q = st.text_input("Ask a follow-up question...", key="chart_q")
        
        if st.button("Ask Gemini (Chart)"):
            if "last_images" not in st.session_state: st.warning("Upload charts first.")
            elif not api_key: st.error("Missing API Key")
            else:
                with st.spinner("Reviewing..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        response = model.generate_content([chart_q] + st.session_state["last_images"])
                        st.session_state["chart_q_response"] = response.text
                    except Exception as e: st.error(f"Error: {e}")

        if st.session_state["chart_q_response"]:
            st.info(st.session_state["chart_q_response"])
            st.download_button("📥 Download Q&A", st.session_state["chart_q_response"], "Gemini_Chart_QA.txt")
