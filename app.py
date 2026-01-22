import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta, datetime
from scipy.stats import norm
from PIL import Image
import io
import google.generativeai as genai
import yfinance as yf
import json
import time 

# Try importing GoogleNews, handle error if not installed
try:
    from GoogleNews import GoogleNews
except ImportError:
    pass # Handle gracefully if missing

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MSTR Command Center",
    page_icon="🚀",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .pass { color: #00FF7F; font-weight: bold; }
    .fail { color: #FF4B4B; font-weight: bold; }
    .warning { color: #FFA500; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

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
#  MATH ENGINE (ALL CALCULATIONS HERE)
# =========================================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_gamma(S, K, T, r, sigma):
    try:
        if T <= 0 or sigma <= 0: return 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except: return 0

def calculate_max_pain(options_chain):
    strikes = options_chain['strike'].unique()
    max_pain_data = []
    for strike in strikes:
        calls_at_strike = options_chain[options_chain['type'] == 'call']
        puts_at_strike = options_chain[options_chain['type'] == 'put']
        call_loss = calls_at_strike.apply(lambda x: max(0, strike - x['strike']) * x['openInterest'], axis=1).sum()
        put_loss = puts_at_strike.apply(lambda x: max(0, x['strike'] - strike) * x['openInterest'], axis=1).sum()
        max_pain_data.append({'strike': strike, 'total_loss': call_loss + put_loss})
    
    df_pain = pd.DataFrame(max_pain_data)
    if df_pain.empty: return 0
    return df_pain.loc[df_pain['total_loss'].idxmin()]['strike']

def calculate_dmi(df, period=14):
    df = df.copy()
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/period).mean() / df['TR'].ewm(alpha=1/period).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/period).mean() / df['TR'].ewm(alpha=1/period).mean())
    df['ADX'] = 100 * abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(alpha=1/period).mean()
    return df

# --- CACHED FUNCTIONS (Prevents Rate Limiting) ---
@st.cache_data(ttl=3600) 
def calculate_dss_data(ticker, period=10, ema_period=9):
    try:
        time.sleep(0.5) 
        df = yf.download(ticker, period="6mo", progress=False)
        if len(df) < period + ema_period: return None
        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        high, low, close = df['High'], df['Low'], df['Close']
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_raw = (close - lowest_low) / (highest_high - lowest_low) * 100
        xPreCalc = stoch_raw.ewm(span=ema_period, adjust=False).mean()
        lowest_smooth = xPreCalc.rolling(window=period).min()
        highest_smooth = xPreCalc.rolling(window=period).max()
        denominator = (highest_smooth - lowest_smooth).replace(0, 1)
        stoch_smooth = (xPreCalc - lowest_smooth) / denominator * 100
        dss = stoch_smooth.ewm(span=ema_period, adjust=False).mean()
        df['DSS'] = dss
        df['Signal'] = df['DSS'].ewm(span=4, adjust=False).mean()
        return df[['Close', 'DSS', 'Signal']].dropna().reset_index()
    except: return None

# =========================================================
#  APP LAYOUT
# =========================================================
st.title("🚀 MSTR Command Center")

# --- SIDEBAR (SETTINGS) ---
st.sidebar.header("🌍 Global Settings")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.markdown("---")
symbol = st.sidebar.text_input("Symbol", value="MSTR").upper()
current_stock_price = st.sidebar.number_input("Current Stock Price ($)", value=158.00, step=0.50)
strike_price = st.sidebar.number_input("Strike Price ($)", value=157.50, step=0.50)
expiration_date = st.sidebar.date_input("Expiration Date", value=date(2026, 1, 9))
purchase_date = st.sidebar.date_input("Purchase Date", value=date.today())
entry_price = st.sidebar.number_input("Entry Price", value=8.55, step=0.10)
implied_volatility = st.sidebar.slider("Implied Volatility (IV %)", 10, 200, 95) / 100.0
risk_free_rate = 0.045
contracts = st.sidebar.number_input("Contracts", value=1, step=1)

# --- SIDEBAR (NAVIGATION) ---
st.sidebar.markdown("---")
st.sidebar.header("📍 Navigation")
# FIXED: Updated menu name to match logic exactly
page = st.sidebar.radio("Go to:", [
    "⚔️ Strategy Battle", 
    "📊 Market Dashboard", 
    "🤖 AI Chart Analysis", 
    "📅 Catalyst & Checklist",
    "📸 Option Chain Visualizer",
    "🧮 Strategy Simulator",
    "⚡ Lambda Analysis",
    "📈 Strike Comparison"
])

# =========================================================
#  PAGE 1: STRATEGY BATTLE
# =========================================================
if page == "⚔️ Strategy Battle":
    st.subheader(f"⚖️ Compare Strategies")
    if st.button("🔄 Reset Scenarios"): st.rerun()
    col_a, col_b = st.columns(2)

    with col_a:
        st.info("### 🔵 Strategy A")
        type_a = st.selectbox("Option Type", ["Call", "Put"], index=0, key="type_a")
        strike_a = st.number_input("Strike ($)", value=strike_price, step=0.50, key="str_a")
        iv_a_display = st.slider("IV % (Scenario A)", 10, 200, int(implied_volatility*100), key="iv_a_slide")
        iv_a = iv_a_display / 100.0
        exp_date_a = st.date_input("Expiration", value=expiration_date, key="exp_a")
        entry_price_a = st.number_input("Entry Price", value=entry_price, step=0.10, key="ent_a")
        contracts_a = st.number_input("Contracts", value=contracts, step=1, key="cnt_a")
        st.markdown("---")
        sim_date_a = st.slider("Sell Date", min_value=date.today(), max_value=exp_date_a, value=date.today() + timedelta(days=5), key="d_a", format="MMM DD")
        sim_price_a = st.slider("Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="p_a")
        days_a = (exp_date_a - sim_date_a).days
        years_a = max(days_a / 365.0, 0.0001)
        opt_price_a = black_scholes(sim_price_a, strike_a, years_a, risk_free_rate, iv_a, type_a.lower())
        profit_a = (opt_price_a * 100 * contracts_a) - (entry_price_a * 100 * contracts_a)
        st.metric(f"Est. {type_a} Value", f"${opt_price_a:.2f}")
        st.metric("Net Profit (A)", f"${profit_a:,.2f}", delta_color="normal" if profit_a >= 0 else "inverse")

    with col_b:
        st.warning("### 🟠 Strategy B")
        type_b = st.selectbox("Option Type", ["Call", "Put"], index=0, key="type_b")
        strike_b = st.number_input("Strike ($)", value=strike_price, step=0.50, key="str_b")
        iv_b_display = st.slider("IV % (Scenario B)", 10, 200, int(implied_volatility*100), key="iv_b_slide")
        iv_b = iv_b_display / 100.0
        exp_date_b = st.date_input("Expiration", value=expiration_date, key="exp_b")
        entry_price_b = st.number_input("Entry Price", value=entry_price, step=0.10, key="ent_b")
        contracts_b = st.number_input("Contracts", value=contracts, step=1, key="cnt_b")
        st.markdown("---")
        sim_date_b = st.slider("Sell Date", min_value=date.today(), max_value=exp_date_b, value=date.today() + timedelta(days=5), key="d_b", format="MMM DD")
        sim_price_b = st.slider("Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="p_b")
        days_b = (exp_date_b - sim_date_b).days
        years_b = max(days_b / 365.0, 0.0001)
        opt_price_b = black_scholes(sim_price_b, strike_b, years_b, risk_free_rate, iv_b, type_b.lower())
        profit_b = (opt_price_b * 100 * contracts_b) - (entry_price_b * 100 * contracts_b)
        st.metric(f"Est. {type_b} Value", f"${opt_price_b:.2f}")
        st.metric("Net Profit (B)", f"${profit_b:,.2f}", delta_color="normal" if profit_b >= 0 else "inverse")

    st.write("---")
    diff = profit_a - profit_b
    if diff > 0: st.success(f"🏆 **Strategy A Wins!** (+${diff:,.2f})")
    elif diff < 0: st.warning(f"🏆 **Strategy B Wins!** (+${abs(diff):,.2f})")
    else: st.info("🤝 Draw")

    data_comp = {"Metric": ["Option Type", "IV %", "Profit"], "Strategy A": [type_a, iv_a_display, profit_a], "Strategy B": [type_b, iv_b_display, profit_b]}
    st.download_button("📥 Download Comparison", pd.DataFrame(data_comp).to_csv().encode('utf-8'), "Comp.csv", "text/csv")
    
    st.markdown("---")
    st.subheader("🗺️ Profit Heatmap")
    map_choice = st.radio("Show Map for:", ["Strategy A 🔵", "Strategy B 🟠"], horizontal=True)
    if map_choice == "Strategy A 🔵": h_st, h_ex, h_en, h_cn, h_type, h_iv = strike_a, exp_date_a, entry_price_a, contracts_a, type_a.lower(), iv_a
    else: h_st, h_ex, h_en, h_cn, h_type, h_iv = strike_b, exp_date_b, entry_price_b, contracts_b, type_b.lower(), iv_b
    
    prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
    future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)]
    heatmap_data = []
    for d in future_dates:
        t_years = max((h_ex - d).days / 365.0, 0.0001)
        for p in prices:
            opt = black_scholes(p, h_st, t_years, risk_free_rate, h_iv, h_type)
            heatmap_data.append({"Date": d.strftime('%Y-%m-%d'), "Stock Price": round(p, 2), "Profit": round((opt - h_en)*100*h_cn, 2)})
    df_hm = pd.DataFrame(heatmap_data)
    c = alt.Chart(df_hm).mark_rect().encode(x='Date:O', y='Stock Price:O', color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)), tooltip=['Date','Stock Price','Profit']).properties(height=350)
    st.altair_chart(c, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📉 Time Decay Comparison")
    decay_data = []
    for i in range(120):
        d = date.today() + timedelta(days=i)
        if d < exp_date_a:
            ta = max((exp_date_a - d).days/365.0, 0.0001)
            val_a = black_scholes(sim_price_a, strike_a, ta, risk_free_rate, iv_a, type_a.lower())
            decay_data.append({"Date": d, "Value": val_a, "Strategy": f"Strategy A ({type_a}) 🔵"})
        if d < exp_date_b:
            tb = max((exp_date_b - d).days/365.0, 0.0001)
            val_b = black_scholes(sim_price_b, strike_b, tb, risk_free_rate, iv_b, type_b.lower())
            decay_data.append({"Date": d, "Value": val_b, "Strategy": f"Strategy B ({type_b}) 🟠"})
    if not decay_data: st.warning("⚠️ No time decay data.")
    else: st.altair_chart(alt.Chart(pd.DataFrame(decay_data)).mark_line(strokeWidth=3).encode(x='Date:T', y='Value:Q', color='Strategy').properties(height=350).interactive(), use_container_width=True)

    st.markdown("---")
    if "compare_ai_response" not in st.session_state: st.session_state["compare_ai_response"] = ""
    comp_q = st.text_input("Ask a question about comparisons...", key="comp_q")
    if st.button("Ask Gemini (Comparison)"):
        if not api_key: st.error("Missing API Key")
        else:
            try:
                genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-2.0-flash')
                st.session_state["compare_ai_response"] = model.generate_content(f"Compare Option A ({type_a}, IV {iv_a*100}%) vs B ({type_b}, IV {iv_b*100}%). User Q: {comp_q}").text
            except Exception as e: st.error(str(e))
    if st.session_state["compare_ai_response"]:
        st.info(st.session_state["compare_ai_response"])

# =========================================================
#  PAGE 2: MARKET DASHBOARD
# =========================================================
elif page == "📊 Market Dashboard":
    st.subheader("📊 DSS Bressert Scanner")
    col_in1, col_in2 = st.columns(2)
    with col_in1: text_tickers = st.text_input("Manual Tickers (comma separated)", value="MSTR, BTC-USD, COIN, NVDA, IBIT, MSTU")
    with col_in2: uploaded_file = st.file_uploader("📂 Upload Excel List", type=['xlsx', 'xls'])

    if "scan_results" not in st.session_state: st.session_state["scan_results"] = []
    
    if st.button("🔎 Scan Market"):
        final_tickers = [t.strip().upper() for t in text_tickers.split(",") if t.strip()]
        if uploaded_file:
            try:
                df_up = pd.read_excel(uploaded_file)
                final_tickers.extend([str(t).strip().upper() for t in df_up.iloc[:, 0].tolist()])
            except: st.error("Error reading Excel")
        final_tickers = list(set(final_tickers))
        
        res = []
        prog = st.progress(0)
        
        for i, t in enumerate(final_tickers):
            df_h = calculate_dss_data(t)
            if df_h is not None and not df_h.empty:
                dss = df_h['DSS'].iloc[-1]
                stat = "🟢 OVERSOLD" if dss <= 20 else "🔴 OVERBOUGHT" if dss >= 80 else "Neutral"
                res.append({"Ticker": t, "Price": f"${df_h['Close'].iloc[-1]:,.2f}", "DSS": round(dss, 2), "Status": stat})
            
            prog.progress((i+1)/len(final_tickers))
            if len(final_tickers) > 10:
                time.sleep(1) 
        
        prog.empty()
        st.session_state["scan_results"] = res

    if st.session_state["scan_results"]:
        df_r = pd.DataFrame(st.session_state["scan_results"])
        def color_s(v): return 'background-color: #d4edda; color: green' if 'OVERSOLD' in v else 'background-color: #f8d7da; color: red' if 'OVERBOUGHT' in v else ''
        st.dataframe(df_r.style.applymap(color_s, subset=['Status']), use_container_width=True)
        st.download_button("📥 Download Scan CSV", df_r.to_csv().encode('utf-8'), "Scan.csv", "text/csv")

        st.write("---")
        sel = st.selectbox("Select Ticker to Chart:", [r['Ticker'] for r in st.session_state["scan_results"]])
        if sel:
            df_c = calculate_dss_data(sel)
            if df_c is not None:
                df_m = df_c.melt('Date', value_vars=['DSS','Signal'], var_name='Line', value_name='Value')
                chart = alt.Chart(df_m).mark_line().encode(x=alt.X('Date:T', title='Date (Daily)'), y=alt.Y('Value', scale=alt.Scale(domain=[0,100])), color=alt.Color('Line', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))).properties(height=350)
                st.altair_chart((chart + alt.Chart(pd.DataFrame({'y':[80]})).mark_rule(color='red').encode(y='y') + alt.Chart(pd.DataFrame({'y':[20]})).mark_rule(color='green').encode(y='y')).interactive(), use_container_width=True)
    
    # --- PUT/CALL RATIO SECTION ---
    st.markdown("---")
    st.subheader("⚖️ Daily Put/Call Ratio (Sentiment)")
    
    pcr_ticker = yf.Ticker(symbol)
    try:
        avail_dates = pcr_ticker.options
        if avail_dates:
            pcr_date = st.selectbox("Select Expiration Date for PCR:", avail_dates, index=0)
            
            if st.button("📊 Calculate PCR"):
                with st.spinner("Fetching Option Chain..."):
                    time.sleep(0.5)
                    chain = pcr_ticker.option_chain(pcr_date)
                    calls = chain.calls
                    puts = chain.puts
                    
                    c_vol = calls['volume'].sum() if not calls.empty else 0
                    p_vol = puts['volume'].sum() if not puts.empty else 0
                    c_oi = calls['openInterest'].sum() if not calls.empty else 0
                    p_oi = puts['openInterest'].sum() if not puts.empty else 0
                    
                    pcr_vol = p_vol / c_vol if c_vol > 0 else 0
                    pcr_oi = p_oi / c_oi if c_oi > 0 else 0
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Call Vol", f"{int(c_vol):,}")
                    c2.metric("Total Put Vol", f"{int(p_vol):,}")
                    
                    # Sentiment Coloring
                    pcr_delta = "off"
                    if pcr_vol > 1.0: pcr_delta = "inverse" # Bearish/High Puts
                    else: pcr_delta = "normal" # Bullish
                    
                    c3.metric("PCR (Volume)", f"{pcr_vol:.2f}", delta="Bearish" if pcr_vol > 1.0 else "Bullish", delta_color=pcr_delta)
                    c4.metric("PCR (Open Int)", f"{pcr_oi:.2f}")
                    
                    if pcr_vol > 1.0:
                        st.warning("⚠️ High Put Volume detected relative to Calls. Sentiment is likely Bearish or Hedging.")
                    elif pcr_vol < 0.7:
                        st.success("🟢 High Call Volume detected relative to Puts. Sentiment appears Bullish.")
                    else:
                        st.info("ℹ️ Put/Call Ratio is neutral (0.7 - 1.0).")
    except: pass

    # --- HIGH VELOCITY SCANNER ---
    st.markdown("---")
    st.subheader("⚡ High-Velocity Option Scanner (Earliest Expiry)")

    if st.button("🎲 Scan Nearest Expiry Chain"):
        try:
            tick = yf.Ticker(symbol)
            dates = tick.options
            
            if not dates:
                st.error(f"No options data found for {symbol}")
            else:
                target_date = dates[0]
                st.info(f"📅 Analyzing Expiry: **{target_date}**")
                time.sleep(0.5)
                
                chain = tick.option_chain(target_date)
                calls = chain.calls.copy()
                calls['Type'] = 'Call'
                puts = chain.puts.copy()
                puts['Type'] = 'Put'
                df = pd.concat([calls, puts], ignore_index=True)
                
                current_price = tick.history(period="1d")['Close'].iloc[-1]
                scanner_data = []
                
                for index, row in df.iterrows():
                    strike = row['strike']
                    vol = row['volume'] if row['volume'] > 0 else 0
                    oi = row['openInterest'] if row['openInterest'] > 0 else 1 
                    iv = row['impliedVolatility']
                    opt_type = row['Type']
                    vol_oi_ratio = vol / oi
                    
                    if opt_type == 'Call':
                        moneyness = ((current_price - strike) / strike) * 100
                    else:
                        moneyness = ((strike - current_price) / current_price) * 100
                    
                    days_to_exp = (pd.to_datetime(target_date) - pd.Timestamp.now()).days
                    t_years = max(days_to_exp / 365.0, 0.001)
                    
                    delta = calculate_delta(current_price, strike, t_years, risk_free_rate, iv, opt_type.lower())
                    gamma = calculate_gamma(current_price, strike, t_years, risk_free_rate, iv)
                    
                    scanner_data.append({
                        "Strike": strike,
                        "Price": f"${current_price:.2f}",
                        "Type": opt_type,
                        "Vol": int(vol),
                        "OI": int(oi),
                        "Vol/OI": round(vol_oi_ratio, 2),
                        "IV": f"{iv*100:.1f}%",
                        "Delta": round(delta, 2),
                        "Gamma": round(gamma, 4),
                        "Moneyness": f"{moneyness:.1f}%"
                    })
                
                df_scan = pd.DataFrame(scanner_data)
                df_scan['Abs_Mon'] = df_scan['Moneyness'].str.strip('%').astype(float).abs()
                df_scan = df_scan[df_scan['Abs_Mon'] < 40].drop(columns=['Abs_Mon'])
                df_scan = df_scan.sort_values(by="Vol/OI", ascending=False).reset_index(drop=True)

                st.write("🔥 **Top Strikes by Volume/OI Velocity** (Smart Money Tracking)")
                
                def highlight_hot(val):
                    if val > 2.0:
                        return 'background-color: #1c4f2a; color: white; font-weight: bold'
                    return ''
                
                st.dataframe(df_scan.style.applymap(highlight_hot, subset=['Vol/OI']), use_container_width=True, height=500)
                
                st.download_button(
                    label="📥 Download Results (Excel/CSV)",
                    data=df_scan.to_csv(index=False).encode('utf-8'),
                    file_name=f"{symbol}_Option_Scan_{target_date}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Scanner Error: {e}")

# =========================================================
#  PAGE 3: AI ANALYST
# =========================================================
elif page == "🤖 AI Chart Analysis":
    st.subheader("🤖 AI Chart Analysis")
    up_files = st.file_uploader("Upload Screenshots...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if "ai_analysis_text" not in st.session_state: st.session_state["ai_analysis_text"] = ""
    if "chart_q_response" not in st.session_state: st.session_state["chart_q_response"] = ""

    if up_files:
        imgs = [Image.open(f) for f in up_files]
        st.session_state["last_images"] = imgs
        cols = st.columns(len(imgs))
        for i, img in enumerate(imgs): cols[i].image(img, use_container_width=True)
        
        if st.button("✨ Analyze All"):
            if api_key:
                try:
                    genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-2.0-flash')
                    st.session_state["ai_analysis_text"] = model.generate_content([f"Analyze {symbol} charts."] + imgs).text
                except Exception as e: st.error(str(e))
        
        if st.session_state["ai_analysis_text"]:
            st.write(st.session_state["ai_analysis_text"])
            st.download_button("📥 Download Report", st.session_state["ai_analysis_text"], "Analysis.txt")

        st.markdown("---")
        cq = st.text_input("Follow-up Question:", key="cq")
        if st.button("Ask Gemini (Chart)"):
            if api_key:
                try:
                    genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-2.0-flash')
                    st.session_state["chart_q_response"] = model.generate_content([cq] + imgs).text
                except Exception as e: st.error(str(e))
        if st.session_state["chart_q_response"]:
            st.info(st.session_state["chart_q_response"])

# =========================================================
#  PAGE 4: CATALYST, DMI & CHECKLIST
# =========================================================
elif page == "📅 Catalyst & Checklist":
    st.markdown("---")
    st.markdown("### 3. 🔴 Real-Time News (Google)")
    
    # Check if GoogleNews is available
    try:
        from GoogleNews import GoogleNews 
        news_client = GoogleNews(period='1d') 
        
        crypto_stocks = ['MSTR', 'COIN', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF', 'IBIT', 'MSTU']
        if symbol in crypto_stocks:
            search_query = f"{symbol} stock bitcoin"
        else:
            search_query = f"{symbol} stock news"
            
        st.write(f"🔎 *Searching for: '{search_query}'*") 
        
        try:
            news_client.search(search_query)
            results = news_client.result()
            
            if results:
                for article in results[:5]: 
                    title = article.get('title')
                    date_posted = article.get('date') 
                    link = article.get('link')
                    media = article.get('media') 
                    
                    if link:
                        with st.expander(f"⏰ {date_posted} | {media}: {title}"):
                            st.write(f"[Read Article]({link})")
            else:
                st.info(f"No recent news found for {symbol}.")
                
        except Exception as e:
            st.error(f"News Error: {e}")
    except ImportError:
        st.warning("GoogleNews library is not installed. News feature disabled.")
  
# =========================================================
#  PAGE 5: OPTION CHAIN VISUALIZER
# =========================================================
elif page == "📸 Option Chain Visualizer":
    st.subheader("📸 Option Chain Visualizer")
    st.markdown("Upload a screenshot of an option chain (Calls on Left, Puts on Right).")
    
    uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Option Chain", use_container_width=True)
        
        if st.button("⚡ Extract & Graph"):
            if not api_key:
                st.error("Missing Gemini API Key")
            else:
                with st.spinner("Gemini is reading the numbers..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        
                        prompt = """
                        Look at this option chain image. 
                        Extract the data for every visible row into a JSON format.
                        Return ONLY a raw JSON list of objects with these keys:
                        [
                          {"strike": 100.0, "call_vol": 500, "put_vol": 200, "call_oi": 1000, "put_oi": 800},
                          ...
                        ]
                        Ensure numbers are integers or floats. If a value is '-', '0', or empty, treat it as 0.
                        Do not include markdown formatting like ```json. Just the raw list.
                        """
                        
                        response = model.generate_content([prompt, image])
                        clean_text = response.text.replace("```json", "").replace("```", "").strip()
                        
                        try:
                            data = json.loads(clean_text)
                            df_chain = pd.DataFrame(data)
                            
                            if not df_chain.empty:
                                st.success(f"Successfully extracted {len(df_chain)} rows!")
                                
                                st.subheader("📊 Volume Battle: Calls vs Puts")
                                df_vol = df_chain.melt(id_vars=['strike'], value_vars=['call_vol', 'put_vol'], var_name='Type', value_name='Volume')
                                df_vol['Type'] = df_vol['Type'].replace({'call_vol': 'Calls 🟢', 'put_vol': 'Puts 🔴'})
                                
                                chart_vol = alt.Chart(df_vol).mark_bar().encode(
                                    x=alt.X('strike:O', title='Strike Price'),
                                    y=alt.Y('Volume:Q'),
                                    color=alt.Color('Type', scale=alt.Scale(domain=['Calls 🟢', 'Puts 🔴'], range=['#00FF00', '#FF0000'])),
                                    tooltip=['strike', 'Type', 'Volume']
                                ).properties(height=400)
                                st.altair_chart(chart_vol, use_container_width=True)
                                
                                st.subheader("🧱 Open Interest Walls")
                                df_oi = df_chain.melt(id_vars=['strike'], value_vars=['call_oi', 'put_oi'], var_name='Type', value_name='Open Interest')
                                df_oi['Type'] = df_oi['Type'].replace({'call_oi': 'Call OI 🔵', 'put_oi': 'Put OI 🟠'})
                                
                                chart_oi = alt.Chart(df_oi).mark_bar().encode(
                                    x=alt.X('strike:O', title='Strike Price'),
                                    y=alt.Y('Open Interest:Q'),
                                    color=alt.Color('Type', scale=alt.Scale(domain=['Call OI 🔵', 'Put OI 🟠'], range=['#1f77b4', '#ff7f0e'])),
                                    tooltip=['strike', 'Type', 'Open Interest']
                                ).properties(height=400)
                                st.altair_chart(chart_oi, use_container_width=True)
                                
                                with st.expander("View Raw Extracted Data"):
                                    st.dataframe(df_chain)
                                    
                            else:
                                st.error("Gemini returned empty data. Please ensure the screenshot is clear.")
                                
                        except json.JSONDecodeError:
                            st.error("Failed to parse Gemini response. Try again.")
                            st.write(response.text) 
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

# =========================================================
#  PAGE 6: STRATEGY SIMULATOR
# =========================================================
elif page == "🧮 Strategy Simulator":
    st.header("🧮 Options Strategy Profit/Loss Calculator")
    
    st.info("Select a strategy below to load the setup guide and calculator.")
    
    # 1. Select Strategy
    strategy = st.selectbox("Select Strategy", [
        "Bull Call Spread (Debit)",
        "Bear Call Spread (Credit)",
        "Bull Put Spread (Credit)",
        "Bear Put Spread (Debit)",
        "Calendar Spread (Long)"
    ])
    
    # 2. Strategy Guide (Parsed from CSV)
    guide_expander = st.expander(f"📘 Guide: {strategy}", expanded=True)
    guide_text = ""
    with guide_expander:
        if "Bull Call" in strategy:
            guide_text = """
            * **Weinstein Stage:** Stage 2 (Up) - Rocket Ship 🚀
            * **Market Look:** Rocket Ship 🚀
            * **Strategy Logic:** Best ROI for catching a strong trend.
            * **Outlook:** Bullish (UP) - I want stock to go UP.
            * **Setup:**
                * Leg 1 (Buy): Low Strike Call (Expensive, Near Stock Price). **Pick Delta ~0.70**
                * Leg 2 (Sell): High Strike Call (Cheap, High Strike). **Delta ~0.30**
            * **IV:** Low (We want it to Rise).
            * **Time Decay:** Hurts me (Negative).
            * **Entry:** Breakout above resistance.
            * **Exit:** Close at +50% profit or if support/resistance break.
            """
        elif "Bear Call" in strategy:
            guide_text = """
            * **Weinstein Stage:** Stage 3 (Top) - Choppy / Nervous
            * **Market Look:** Choppy / Nervous
            * **Strategy Logic:** Safe way to bet "the party is over" without needing a crash yet.
            * **Outlook:** Bearish/Flat - I want stock to go DOWN or stay Flat.
            * **Setup:**
                * Leg 1 (Sell): Low Strike Call (Expensive, Near Stock Price). **Pick Delta ~0.30**
                * Leg 2 (Buy): High Strike Call (Cheap, High Strike). **Delta ~0.10**
            * **IV:** High (We want it to Drop).
            * **Time Decay:** Helps me (Positive).
            * **Entry:** Reversal Candle (outside Bar) at support/Resistance.
            * **Exit:** Not specified (Aim for credit decay).
            """
        elif "Bull Put" in strategy:
            guide_text = """
            * **Weinstein Stage:** Stage 2 (Bullish/Support) - Uptrend
            * **Outlook:** Bullish/Flat - I want stock to go UP or stay Flat.
            * **Setup:**
                * Leg 1 (Sell): High Strike Put (Expensive, Near Stock Price). **Pick Delta ~0.30**
                * Leg 2 (Buy): Low Strike Put (Cheap, Low Strike). **Delta ~0.10**
            * **IV:** High (We want it to Drop).
            * **Time Decay:** Helps me (Positive).
            * **Entry:** Reversal Candle (outside Bar) at support/Resistance.
            * **Exit:** Close at +50% profit.
            """
        elif "Bear Put" in strategy:
            guide_text = """
            * **Weinstein Stage:** Stage 4 (Down) - Waterfall 📉
            * **Market Look:** Waterfall 📉
            * **Strategy Logic:** Captures the panic selling for big gains.
            * **Outlook:** Bearish/Down - I want stock to go DOWN.
            * **Setup:**
                * Leg 1 (Buy): High Strike Put (Expensive, Near Stock Price). **Pick Delta ~0.70**
                * Leg 2 (Sell): Low Strike Put (Cheap, Low Strike). **Delta ~0.30**
            * **IV:** Low (We want it to Rise).
            * **Time Decay:** Hurts me (Negative).
            * **Entry:** Breakdown below support.
            * **Exit:** Close at +50% profit or if support/resistance break.
            """
        elif "Calendar" in strategy:
            guide_text = """
            * **Weinstein Stage:** Stage 1 (Base) - Flat / Boring
            * **Market Look:** Flat / Boring
            * **Strategy Logic:** You profit if the stock does absolutely nothing.
            * **Outlook:** Neutral / Stock Stay - I want stock to Sit Still/Stagnate.
            * **Setup:**
                * Leg 1 (Sell): Near Expiration. **Delta ~0.50**
                * Leg 2 (Buy): Far Expiration. **Delta ~0.50**
                * Same Strike.
            * **IV:** Low (We want it to Rise).
            * **Time Decay:** Helps me (Positive).
            * **Entry:** Inside Bar (quiet/coiled).
            * **Exit:** Close before earnings. Don't hold through event.
            """
        st.markdown(guide_text)

    st.markdown("---")
    
    # 3. Calculator Inputs
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        sim_price = st.number_input("Current Stock Price ($)", value=current_stock_price)
    with col_main2:
        sim_qty = st.number_input("Number of Spreads", value=1, min_value=1)
        
    st.subheader("Leg Configuration")
    c1, c2 = st.columns(2)
    
    # Initialize session state for inputs if needed
    if "leg1_price" not in st.session_state: st.session_state.leg1_price = 10.0
    if "leg2_price" not in st.session_state: st.session_state.leg2_price = 5.0

    # Logic to label legs based on strategy
    report_data = {}
    if "Calendar" in strategy:
        # Calendar: Same Strike, Diff Exp
        with c1:
            st.markdown("### 🦵 Leg 1 (Sell/Near)")
            k1 = st.number_input("Strike Price", value=sim_price)
            t1_date = st.date_input("Expiration (Near)", value=date.today() + timedelta(days=30))
            p1 = st.number_input("Premium (Price)", value=st.session_state.leg1_price, key="p1")
            lbl1 = "Sell (Near)"
            
        with c2:
            st.markdown("### 🦵 Leg 2 (Buy/Far)")
            # Strike is same usually
            st.info(f"Strike: ${k1}") 
            t2_date = st.date_input("Expiration (Far)", value=date.today() + timedelta(days=60))
            p2 = st.number_input("Premium (Price)", value=st.session_state.leg2_price, key="p2")
            lbl2 = "Buy (Far)"
            k2 = k1 # For logic consistency
            
        # Helper to estimate price
        if st.button("🔮 Estimate Option Prices (Black-Scholes)"):
            t1_y = (t1_date - date.today()).days / 365.0
            t2_y = (t2_date - date.today()).days / 365.0
            est_p1 = black_scholes(sim_price, k1, t1_y, risk_free_rate, implied_volatility, 'call') # Assume call calendar
            est_p2 = black_scholes(sim_price, k1, t2_y, risk_free_rate, implied_volatility, 'call')
            st.session_state.leg1_price = round(est_p1, 2)
            st.session_state.leg2_price = round(est_p2, 2)
            st.rerun()
        report_data = {"Strategy": strategy, "Strike": k1, "Short Exp": str(t1_date), "Long Exp": str(t2_date), "Short Prem": p1, "Long Prem": p2}
            
    else:
        # Vertical Spreads
        is_call = "Call" in strategy
        if "Bull Call" in strategy:
            lbl1 = "Buy (Low Strike, Expensive)"
            lbl2 = "Sell (High Strike, Cheap)"
            def_k1 = sim_price - 5 # ITM
            def_k2 = sim_price + 5 # OTM
            is_debit = True
        elif "Bear Call" in strategy:
            lbl1 = "Sell (Low Strike, Expensive)"
            lbl2 = "Buy (High Strike, Cheap)"
            def_k1 = sim_price - 5 # ITM
            def_k2 = sim_price + 5 # OTM
            is_debit = False
        elif "Bull Put" in strategy:
            lbl1 = "Sell (High Strike, Expensive)"
            lbl2 = "Buy (Low Strike, Cheap)"
            def_k1 = sim_price + 5 # ITM Put
            def_k2 = sim_price - 5 # OTM Put
            is_debit = False
        elif "Bear Put" in strategy:
            lbl1 = "Buy (High Strike, Expensive)"
            lbl2 = "Sell (Low Strike, Cheap)"
            def_k1 = sim_price + 5 # ITM Put
            def_k2 = sim_price - 5 # OTM Put
            is_debit = True
        
        with c1:
            st.markdown(f"### 🦵 Leg 1: {lbl1}")
            k1 = st.number_input("Strike 1 ($)", value=float(def_k1))
            p1 = st.number_input("Premium ($)", value=st.session_state.leg1_price, key="p1_v")
            
        with c2:
            st.markdown(f"### 🦵 Leg 2: {lbl2}")
            k2 = st.number_input("Strike 2 ($)", value=float(def_k2))
            p2 = st.number_input("Premium ($)", value=st.session_state.leg2_price, key="p2_v")

        if st.button("🔮 Estimate Prices"):
            # Simple assumption: 30 days out
            ty = 30 / 365.0
            o_type = 'call' if is_call else 'put'
            est_p1 = black_scholes(sim_price, k1, ty, risk_free_rate, implied_volatility, o_type)
            est_p2 = black_scholes(sim_price, k2, ty, risk_free_rate, implied_volatility, o_type)
            st.session_state.leg1_price = round(est_p1, 2)
            st.session_state.leg2_price = round(est_p2, 2)
            st.rerun()
        report_data = {"Strategy": strategy, "Leg 1 Type": lbl1, "Strike 1": k1, "Prem 1": p1, "Leg 2 Type": lbl2, "Strike 2": k2, "Prem 2": p2}

    st.markdown("---")
    
    # 4. Calculate P&L
    if st.button("🚀 Calculate Profit/Loss"):
        # Range of prices to simulate (±20%)
        sim_prices = np.linspace(sim_price * 0.8, sim_price * 1.2, 50)
        pnl_data = []
        
        # --- CALCULATION LOGIC ---
        if "Calendar" in strategy:
            # Calendar Logic (Approximate)
            # Net Debit: Buy (Leg 2) - Sell (Leg 1)
            cost = (p2 - p1) * 100 * sim_qty
            
            # Time diff for Long option when Short expires
            dt_near = (t1_date - date.today()).days / 365.0
            dt_far = (t2_date - date.today()).days / 365.0
            remaining_time = dt_far - dt_near
            
            for s in sim_prices:
                # Value of Short at Expiry (Intrinsic)
                # Assume Call Calendar
                val_short = max(0, s - k1) 
                
                # Value of Long at Short Expiry (Black Scholes estimate)
                val_long = black_scholes(s, k1, remaining_time, risk_free_rate, implied_volatility, 'call')
                
                spread_val_at_expiry = (val_long - val_short) * 100 * sim_qty
                profit = spread_val_at_expiry - cost
                pnl_data.append({"Price": s, "P&L": profit})
                
        else:
            # Vertical Logic
            if is_debit:
                # Leg 1 is Buy, Leg 2 is Sell
                net_cost = (p1 - p2) * 100 * sim_qty
            else:
                # Leg 1 is Sell, Leg 2 is Buy
                net_credit = (p1 - p2) * 100 * sim_qty 
                
            for s in sim_prices:
                # Calculate Intrinsic Values at Expiry
                if "Bull Call" in strategy:
                    # Buy Low (K1), Sell High (K2)
                    val_l = max(0, s - k1)
                    val_s = max(0, s - k2)
                    payoff = (val_l - val_s) * 100 * sim_qty
                    profit = payoff - net_cost
                elif "Bear Put" in strategy:
                    # Buy High (K1), Sell Low (K2) (Puts)
                    val_l = max(0, k1 - s)
                    val_s = max(0, k2 - s)
                    payoff = (val_l - val_s) * 100 * sim_qty
                    profit = payoff - net_cost
                elif "Bear Call" in strategy:
                    # Sell Low (K1), Buy High (K2) (Calls)
                    val_short = max(0, s - k1)
                    val_long = max(0, s - k2)
                    loss_on_spread = (val_short - val_long) * 100 * sim_qty
                    profit = net_credit - loss_on_spread
                elif "Bull Put" in strategy:
                    # Sell High (K1), Buy Low (K2) (Puts)
                    val_short = max(0, k1 - s)
                    val_long = max(0, k2 - s)
                    loss_on_spread = (val_short - val_long) * 100 * sim_qty
                    profit = net_credit - loss_on_spread
                    
                pnl_data.append({"Price": s, "P&L": profit})

        # --- DISPLAY RESULTS ---
        df_pnl = pd.DataFrame(pnl_data)
        
        # Metrics
        max_p = df_pnl['P&L'].max()
        max_l = df_pnl['P&L'].min()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Max Profit", f"${max_p:,.2f}")
        m2.metric("Max Loss", f"${max_l:,.2f}")
        
        # Chart
        c = alt.Chart(df_pnl).mark_area(
            line={'color':'white'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#FF4B4B', offset=0),
                       alt.GradientStop(color='#FF4B4B', offset=0.5), 
                       alt.GradientStop(color='#00FF7F', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Price', title='Stock Price at Expiry'),
            y=alt.Y('P&L', title='Profit/Loss ($)'),
            tooltip=['Price', 'P&L']
        ).properties(height=400)
        
        # Add a zero line
        rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='white').encode(y='y')
        
        st.altair_chart(c + rule, use_container_width=True)
        
        # --- DOWNLOAD BUTTONS ---
        st.write("### 💾 Export Analysis")
        col_d1, col_d2 = st.columns(2)
        
        # 1. Download CSV (Excel)
        with col_d1:
            st.download_button(
                label="📥 Download Data (Excel/CSV)",
                data=df_pnl.to_csv(index=False).encode('utf-8'),
                file_name=f"{strategy}_Analysis.csv",
                mime="text/csv"
            )
            
        # 2. Download Report (Word/Doc)
        with col_d2:
            html_report = f"""
            <html>
            <head><title>Strategy Report</title></head>
            <body>
                <h1>Strategy Report: {strategy}</h1>
                <p><b>Date:</b> {date.today()}</p>
                <p><b>Ticker:</b> {symbol}</p>
                <hr>
                <h3>Configuration</h3>
                <ul>
            """
            for k, v in report_data.items():
                html_report += f"<li><b>{k}:</b> {v}</li>"
            
            html_report += f"""
                </ul>
                <hr>
                <h3>Outcome</h3>
                <p><b>Max Profit:</b> ${max_p:,.2f}</p>
                <p><b>Max Loss:</b> ${max_l:,.2f}</p>
                <hr>
                <h3>Strategy Guide</h3>
                {guide_text.replace('*', '').replace('\n', '<br>')}
            </body>
            </html>
            """
            
            st.download_button(
                label="📥 Download Report (Word/Doc)",
                data=html_report,
                file_name=f"{strategy}_Report.doc",
                mime="application/msword"
            )

# =========================================================
#  PAGE 7: LAMBDA ANALYSIS
# =========================================================
elif page == "⚡ Lambda Analysis":
    st.header("⚡ Lambda (Option Leverage) Analyzer")
    st.info("The Indicator for 'Winning Amount vs Premium' per option.")
    
    st.markdown(r"""
    **Lambda ($\lambda$)**, also known as **Gearing**, measures the leverage of an option. 
    It tells you the percentage change in the option price for a 1% change in the underlying stock price.
    
    $$\text{Lambda} = \frac{\text{Stock Price}}{\text{Option Price}} \times \text{Delta}$$
    """)
    
    # Calculate Time to Expiry
    days_to_exp = (expiration_date - date.today()).days
    T_years = max(days_to_exp / 365.0, 0.0001)
    
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        st.subheader("🟢 Call Option Leverage")
        call_price = black_scholes(current_stock_price, strike_price, T_years, risk_free_rate, implied_volatility, 'call')
        call_delta = calculate_delta(current_stock_price, strike_price, T_years, risk_free_rate, implied_volatility, 'call')
        
        if call_price > 0.01:
            call_lambda = (current_stock_price / call_price) * call_delta
            st.metric("Call Lambda ($\lambda$)", f"{call_lambda:.2f}x", help="A 1% rise in stock price = This % rise in Option Price")
            st.write(f"If {symbol} moves **+1%**, this Call moves approx **+{call_lambda:.2f}%**")
        else:
            st.error("Option Price too low to calculate Lambda")

    with col_l2:
        st.subheader("🔴 Put Option Leverage")
        put_price = black_scholes(current_stock_price, strike_price, T_years, risk_free_rate, implied_volatility, 'put')
        put_delta = calculate_delta(current_stock_price, strike_price, T_years, risk_free_rate, implied_volatility, 'put')
        
        if put_price > 0.01:
            put_lambda = (current_stock_price / put_price) * put_delta
            st.metric("Put Lambda ($\lambda$)", f"{put_lambda:.2f}x", help="A 1% drop in stock price = This % rise in Option Price")
            st.write(f"If {symbol} moves **-1%**, this Put moves approx **+{abs(put_lambda):.2f}%**")
        else:
            st.error("Option Price too low to calculate Lambda")
            
    st.markdown("---")
    st.subheader("📈 Leverage Heatmap (Lambda vs Strike)")
    
    # Generate Data for Plot
    strikes = np.linspace(current_stock_price * 0.7, current_stock_price * 1.3, 20)
    lambda_data = []
    
    for k in strikes:
        # Call
        c_p = black_scholes(current_stock_price, k, T_years, risk_free_rate, implied_volatility, 'call')
        c_d = calculate_delta(current_stock_price, k, T_years, risk_free_rate, implied_volatility, 'call')
        c_l = (current_stock_price / c_p * c_d) if c_p > 0.01 else 0
        
        # Put
        p_p = black_scholes(current_stock_price, k, T_years, risk_free_rate, implied_volatility, 'put')
        p_d = calculate_delta(current_stock_price, k, T_years, risk_free_rate, implied_volatility, 'put')
        p_l = (current_stock_price / p_p * p_d) if p_p > 0.01 else 0
        
        lambda_data.append({"Strike": k, "Type": "Call Lambda 🟢", "Value": c_l})
        lambda_data.append({"Strike": k, "Type": "Put Lambda 🔴", "Value": p_l})
        
    df_lam = pd.DataFrame(lambda_data)
    
    c_lam = alt.Chart(df_lam).mark_line(point=True).encode(
        x=alt.X('Strike', title='Strike Price'),
        y=alt.Y('Value', title='Lambda (Leverage Factor)'),
        color='Type',
        tooltip=['Strike', 'Type', 'Value']
    ).interactive()
    
    st.altair_chart(c_lam, use_container_width=True)
    st.caption("Note: Out-of-the-money options (High strikes for Calls, Low for Puts) have higher leverage/Lambda, but lower probability of profit.")

# =========================================================
#  PAGE 8: STRIKE COMPARISON
# =========================================================
elif page == "📈 Strike Comparison":
    st.header("📈 Strike Price History Comparison")
    st.info("Compare the actual historical price action of up to 4 different strike prices.")
    
    # 1. Inputs
    c_comp1, c_comp2, c_comp3 = st.columns(3)
    
    ticker_obj = yf.Ticker(symbol)
    try:
        all_dates = ticker_obj.options
        if not all_dates:
            st.error("No option dates found for this symbol.")
            st.stop()
            
        with c_comp1:
            comp_exp = st.selectbox("1. Expiration Date", all_dates, index=0, key="comp_exp")
            
        with c_comp2:
            comp_type = st.radio("2. Option Type", ["Call", "Put"], horizontal=True, key="comp_type")
            
        # Get Strikes for selected date/type
        chain = ticker_obj.option_chain(comp_exp)
        if comp_type == "Call":
            df_chain = chain.calls
        else:
            df_chain = chain.puts
            
        available_strikes = sorted(df_chain['strike'].tolist())
        
        with c_comp3:
            selected_strikes = st.multiselect("3. Select Strikes (Max 4)", available_strikes, max_selections=4)
            
        # 2. Fetch Data Button
        if st.button("📉 Compare Strike Histories"):
            if not selected_strikes:
                st.warning("Please select at least one strike price.")
            else:
                st.write(f"Fetching history for {len(selected_strikes)} contracts...")
                progress_bar = st.progress(0)
                
                # Dictionary to hold dataframes
                hist_data = {}
                
                for idx, strike in enumerate(selected_strikes):
                    # Find contract symbol
                    contract_row = df_chain[df_chain['strike'] == strike]
                    if not contract_row.empty:
                        contract_symbol = contract_row.iloc[0]['contractSymbol']
                        
                        # Download history
                        try:
                            opt_hist = yf.download(contract_symbol, period="1mo", progress=False)
                            if not opt_hist.empty:
                                # Ensure index is reset so Date is a column
                                opt_hist = opt_hist.reset_index()
                                
                                # FIX: Handle new yfinance MultiIndex column structure
                                # Sometimes columns are like ('Close', 'MSTR260109C00150000')
                                if isinstance(opt_hist.columns, pd.MultiIndex):
                                    opt_hist.columns = opt_hist.columns.get_level_values(0)
                                    
                                # Ensure Date column exists
                                if 'Date' not in opt_hist.columns and 'Datetime' in opt_hist.columns:
                                    opt_hist = opt_hist.rename(columns={'Datetime': 'Date'})
                                
                                # Keep only Date and Close
                                sub_df = opt_hist[['Date', 'Close']].copy()
                                
                                # Force Date to datetime for Altair
                                sub_df['Date'] = pd.to_datetime(sub_df['Date'])
                                
                                sub_df['Strike'] = f"${strike} {comp_type}"
                                sub_df.rename(columns={'Close': 'Price'}, inplace=True)
                                hist_data[strike] = sub_df
                            else:
                                st.warning(f"No volume/history found for {contract_symbol}")
                        except Exception as e:
                            st.warning(f"Error fetching {contract_symbol}: {e}")
                    
                    # Rate limit pause
                    time.sleep(0.3)
                    progress_bar.progress((idx + 1) / len(selected_strikes))
                
                progress_bar.empty()
                
                # Combine Data
                if hist_data:
                    combined_df = pd.concat(hist_data.values(), ignore_index=True)
                    
                    # Chart
                    chart = alt.Chart(combined_df).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Price:Q', title='Option Price ($)'),
                        color=alt.Color('Strike:N', title='Contract'),
                        tooltip=['Date', 'Strike', 'Price']
                    ).properties(height=500).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Show raw data
                    with st.expander("View Raw Price Data"):
                        st.dataframe(combined_df)
                else:
                    st.error("No historical data could be retrieved for selected strikes.")
                    
    except Exception as e:
        st.error(f"Error loading option chain: {e}")
