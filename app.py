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
#  HELPER FUNCTIONS (Math & Data)
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

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_max_pain(options_chain):
    """Calculates the strike price where option writers lose the least money."""
    strikes = options_chain['strike'].unique()
    max_pain_data = []
    for strike in strikes:
        calls_at_strike = options_chain[options_chain['type'] == 'call']
        puts_at_strike = options_chain[options_chain['type'] == 'put']
        # Simple calculation: Sum of Intrinsic Value * Open Interest
        call_loss = calls_at_strike.apply(lambda x: max(0, strike - x['strike']) * x['openInterest'], axis=1).sum()
        put_loss = puts_at_strike.apply(lambda x: max(0, x['strike'] - strike) * x['openInterest'], axis=1).sum()
        max_pain_data.append({'strike': strike, 'total_loss': call_loss + put_loss})
    
    df_pain = pd.DataFrame(max_pain_data)
    if df_pain.empty: return 0
    return df_pain.loc[df_pain['total_loss'].idxmin()]['strike']

def calculate_dss_data(ticker, period=10, ema_period=9):
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        if len(df) < period + ema_period: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
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
st.title("🚀 MSTR Option Command Center")

# --- GLOBAL SIDEBAR ---
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
purchase_date = st.sidebar.date_input("Purchase Date", value=date(2024, 12, 24))
entry_price = st.sidebar.number_input("Entry Price", value=8.55, step=0.10)
implied_volatility = st.sidebar.slider("Implied Volatility (IV %)", 10, 200, 95) / 100.0
risk_free_rate = 0.045
contracts = st.sidebar.number_input("Contracts", value=1, step=1)

# --- 5 TABS ---
tab_math, tab_dashboard, tab_ai, tab_catalyst, tab_deep_dive = st.tabs([
    "⚔️ Strategy Battle", 
    "📊 Market Dashboard", 
    "📸 Chart Analyst", 
    "📅 Catalyst & Checklist",
    "🩻 Deep Dive Validator"
])

# =========================================================
#  TAB 1: STRATEGY BATTLE
# =========================================================
with tab_math:
    st.subheader(f"⚖️ Compare Strategies")
    if st.button("🔄 Reset Scenarios"): st.rerun()
    col_a, col_b = st.columns(2)

    with col_a:
        st.info("### 🔵 Strategy A")
        strike_a = st.number_input("Strike ($)", value=strike_price, step=0.50, key="str_a")
        exp_date_a = st.date_input("Expiration", value=expiration_date, key="exp_a")
        entry_price_a = st.number_input("Entry Price", value=entry_price, step=0.10, key="ent_a")
        contracts_a = st.number_input("Contracts", value=contracts, step=1, key="cnt_a")
        st.markdown("---")
        sim_date_a = st.slider("Sell Date", min_value=date.today(), max_value=exp_date_a, value=date.today() + timedelta(days=5), key="d_a", format="MMM DD")
        sim_price_a = st.slider("Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="p_a")
        
        days_a = (exp_date_a - sim_date_a).days
        years_a = max(days_a / 365.0, 0.0001)
        opt_price_a = black_scholes(sim_price_a, strike_a, years_a, risk_free_rate, implied_volatility)
        profit_a = (opt_price_a * 100 * contracts_a) - (entry_price_a * 100 * contracts_a)
        st.metric("Est. Value", f"${opt_price_a:.2f}")
        st.metric("Net Profit (A)", f"${profit_a:,.2f}", delta_color="normal" if profit_a >= 0 else "inverse")

    with col_b:
        st.warning("### 🟠 Strategy B")
        strike_b = st.number_input("Strike ($)", value=strike_price, step=0.50, key="str_b")
        exp_date_b = st.date_input("Expiration", value=expiration_date, key="exp_b")
        entry_price_b = st.number_input("Entry Price", value=entry_price, step=0.10, key="ent_b")
        contracts_b = st.number_input("Contracts", value=contracts, step=1, key="cnt_b")
        st.markdown("---")
        sim_date_b = st.slider("Sell Date", min_value=date.today(), max_value=exp_date_b, value=date.today() + timedelta(days=5), key="d_b", format="MMM DD")
        sim_price_b = st.slider("Stock Price", min_value=float(current_stock_price * 0.5), max_value=float(current_stock_price * 2.0), value=float(current_stock_price), step=1.0, key="p_b")
        
        days_b = (exp_date_b - sim_date_b).days
        years_b = max(days_b / 365.0, 0.0001)
        opt_price_b = black_scholes(sim_price_b, strike_b, years_b, risk_free_rate, implied_volatility)
        profit_b = (opt_price_b * 100 * contracts_b) - (entry_price_b * 100 * contracts_b)
        st.metric("Est. Value", f"${opt_price_b:.2f}")
        st.metric("Net Profit (B)", f"${profit_b:,.2f}", delta_color="normal" if profit_b >= 0 else "inverse")

    st.write("---")
    diff = profit_a - profit_b
    if diff > 0: st.success(f"🏆 **Strategy A Wins!** (+${diff:,.2f})")
    elif diff < 0: st.warning(f"🏆 **Strategy B Wins!** (+${abs(diff):,.2f})")
    else: st.info("🤝 Draw")

    # Downloads & Heatmap
    data_comp = {"Metric": ["Profit"], "Strategy A": [profit_a], "Strategy B": [profit_b]}
    st.download_button("📥 Download Comparison", pd.DataFrame(data_comp).to_csv().encode('utf-8'), "Comp.csv", "text/csv")
    
    st.markdown("---")
    st.subheader("🗺️ Profit Heatmap")
    map_choice = st.radio("Show Map for:", ["Strategy A 🔵", "Strategy B 🟠"], horizontal=True)
    if map_choice == "Strategy A 🔵": h_st, h_ex, h_en, h_cn = strike_a, exp_date_a, entry_price_a, contracts_a
    else: h_st, h_ex, h_en, h_cn = strike_b, exp_date_b, entry_price_b, contracts_b
    
    prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
    future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)]
    heatmap_data = []
    for d in future_dates:
        t_years = max((h_ex - d).days / 365.0, 0.0001)
        for p in prices:
            opt = black_scholes(p, h_st, t_years, risk_free_rate, implied_volatility)
            heatmap_data.append({"Date": d.strftime('%Y-%m-%d'), "Stock Price": round(p, 2), "Profit": round((opt - h_en)*100*h_cn, 2)})
            
    df_hm = pd.DataFrame(heatmap_data)
    c = alt.Chart(df_hm).mark_rect().encode(x='Date:O', y='Stock Price:O', color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)), tooltip=['Date','Stock Price','Profit']).properties(height=350)
    st.altair_chart(c, use_container_width=True)
    st.download_button("📥 Download Heatmap", df_hm.to_csv().encode('utf-8'), "Heatmap.csv", "text/csv")
    
    # Time Decay
    st.markdown("---")
    st.subheader("📉 Time Decay Comparison")
    decay_data = []
    for i in range(120):
        d = date.today() + timedelta(days=i)
        if d < exp_date_a:
            ta = max((exp_date_a - d).days/365.0, 0.0001)
            decay_data.append({"Date": d, "Value": black_scholes(sim_price_a, strike_a, ta, risk_free_rate, implied_volatility), "Strategy": "Strategy A 🔵"})
        if d < exp_date_b:
            tb = max((exp_date_b - d).days/365.0, 0.0001)
            decay_data.append({"Date": d, "Value": black_scholes(sim_price_b, strike_b, tb, risk_free_rate, implied_volatility), "Strategy": "Strategy B 🟠"})
    st.altair_chart(alt.Chart(pd.DataFrame(decay_data)).mark_line(strokeWidth=3).encode(x='Date:T', y='Value:Q', color='Strategy').properties(height=350).interactive(), use_container_width=True)

    # Q&A
    st.markdown("---")
    if "compare_ai_response" not in st.session_state: st.session_state["compare_ai_response"] = ""
    comp_q = st.text_input("Ask a question about comparisons...", key="comp_q")
    if st.button("Ask Gemini (Comparison)"):
        if not api_key: st.error("Missing API Key")
        else:
            try:
                genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-2.0-flash')
                st.session_state["compare_ai_response"] = model.generate_content(f"Compare Option A (Profit {profit_a}) vs B (Profit {profit_b}). User Q: {comp_q}").text
            except Exception as e: st.error(str(e))
    if st.session_state["compare_ai_response"]:
        st.info(st.session_state["compare_ai_response"])
        st.download_button("📥 Download Explanation", st.session_state["compare_ai_response"], "Battle_QA.txt")

# =========================================================
#  TAB 2: MARKET DASHBOARD
# =========================================================
with tab_dashboard:
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
                chart = alt.Chart(df_m).mark_line().encode(x='Date:T', y=alt.Y('Value', scale=alt.Scale(domain=[0,100])), color=alt.Color('Line', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))).properties(height=350)
                st.altair_chart((chart + alt.Chart(pd.DataFrame({'y':[80]})).mark_rule(color='red').encode(y='y') + alt.Chart(pd.DataFrame({'y':[20]})).mark_rule(color='green').encode(y='y')).interactive(), use_container_width=True)

# =========================================================
#  TAB 3: AI ANALYST
# =========================================================
with tab_ai:
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
            st.download_button("📥 Download Q&A", st.session_state["chart_q_response"], "Chart_QA.txt")

# =========================================================
#  TAB 4: CATALYST & CHECKLIST
# =========================================================
with tab_catalyst:
    st.subheader("📅 Catalyst & Volatility Checker")
    cat_sym = st.text_input("Enter Symbol to Check:", value=symbol, key="cat_sym")
    
    if st.button("🔎 Check Catalysts"):
        tick = yf.Ticker(cat_sym)
        st.write("---")
        
        # 1. EARNINGS CHECK
        st.markdown("### 1. Earnings Calendar")
        try:
            cal = tick.calendar
            if cal is not None and not cal.empty:
                next_earnings = cal.iloc[0][0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
                if next_earnings:
                    earning_date = pd.to_datetime(next_earnings).date()
                    days_left = (earning_date - date.today()).days
                    c1, c2 = st.columns(2)
                    c1.metric("Next Earnings", earning_date.strftime('%Y-%m-%d'))
                    c2.metric("Days Left", f"{days_left} Days")
                    if 0 <= days_left <= 7: st.error("⚠️ **HIGH VOLATILITY WARNING:** Earnings Imminent!")
                    elif days_left < 0: st.info("Earnings just passed.")
                    else: st.success("✅ Earnings are safe distance away.")
                else: st.info("No earnings date found.")
            else: st.info("Earnings data unavailable.")
        except: st.warning("Could not retrieve earnings.")

        # 2. NEWS CHECK
        st.markdown("---")
        st.markdown("### 2. Recent News")
        try:
            news = tick.news
            if news:
                for n in news[:3]:
                    with st.expander(f"📰 {n['title']}"): st.write(f"Source: {n['publisher']} | [Link]({n['link']})")
            else: st.info("No news found.")
        except: st.info("News feed unavailable.")

    # 3. IMPLIED VOLATILITY (IV) CHECKER
    st.markdown("---")
    st.subheader("🔎 Option Implied Volatility (IV)")
    st.write(f"Fetch the **REAL Market IV** for your contract: **Strike ${strike_price}**")
    
    if st.button("📊 Get Market IV"):
        tick = yf.Ticker(cat_sym)
        try:
            avail_dates = tick.options
            if not avail_dates:
                st.error("No option chain data available for this symbol.")
            else:
                target_date_str = expiration_date.strftime('%Y-%m-%d')
                if target_date_str in avail_dates:
                    selected_date = target_date_str
                    msg = f"Found exact expiration: {selected_date}"
                else:
                    target_dt = datetime.strptime(target_date_str, '%Y-%m-%d').date()
                    closest_date = min(avail_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - target_dt))
                    selected_date = closest_date
                    msg = f"⚠️ Target date {target_date_str} not found. Using closest: **{selected_date}**"

                st.info(msg)
                opt_chain = tick.option_chain(selected_date)
                calls = opt_chain.calls
                specific_contract = calls[calls['strike'] == strike_price]
                
                if not specific_contract.empty:
                    row = specific_contract.iloc[0]
                    market_iv = row['impliedVolatility']
                    volume = row['volume']
                    oi = row['openInterest']
                    last_price = row['lastPrice']
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Market IV", f"{market_iv * 100:.2f}%")
                    m2.metric("Last Price", f"${last_price:.2f}")
                    m3.metric("Volume", f"{int(volume) if not pd.isna(volume) else 0}")
                    m4.metric("Open Interest", f"{int(oi) if not pd.isna(oi) else 0}")
                    diff_iv = (market_iv - implied_volatility) * 100
                    st.write("---")
                    if abs(diff_iv) < 5: st.success("✅ IV Matches.")
                    elif diff_iv > 0: st.warning(f"⚠️ Market IV is {diff_iv:.1f}% higher.")
                    else: st.info(f"ℹ️ Market IV is {abs(diff_iv):.1f}% lower.")
                else:
                    st.warning(f"Strike Price ${strike_price} not found.")
        except Exception as e:
            st.error(f"Error fetching option data: {e}")

# =========================================================
#  TAB 5: DEEP DIVE VALIDATOR (NEW!)
# =========================================================
with tab_deep_dive:
    st.header(f"🩻 Deep Dive Validator: {symbol}")
    st.markdown("Advanced 7-Point Checklist using detailed market mechanics.")
    
    # Extra input for this specific tab
    target_profit_move = st.number_input("Target Stock Move ($)", value=2.0, help="How many dollars do you need the stock to move today?", key="dd_move")
    
    if st.button("🚀 Run Deep Analysis"):
        try:
            with st.spinner(f'Deep diving into {symbol}...'):
                tick = yf.Ticker(symbol)
                hist = tick.history(period="5d")
                info = tick.info
                curr_p = hist['Close'].iloc[-1]
                prev_c = hist['Close'].iloc[-2]
                
                # Fetch Options for Tab 5 Logic
                avail_dates = tick.options
                if not avail_dates:
                    st.error("No options found.")
                else:
                    # Auto-select closest date to global expiration
                    t_dt = datetime.strptime(expiration_date.strftime('%Y-%m-%d'), '%Y-%m-%d').date()
                    sel_date = min(avail_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - t_dt))
                    
                    chain = tick.option_chain(sel_date)
                    calls = chain.calls
                    calls['type'] = 'call'
                    puts = chain.puts
                    puts['type'] = 'put'
                    full_chain = pd.concat([calls, puts])
                    
                    # Find specific contract
                    spec_con = calls.iloc[(calls['strike'] - strike_price).abs().argsort()[:1]]
                    
                    if spec_con.empty:
                        st.warning("Strike not found.")
                    else:
                        row = spec_con.iloc[0]
                        c_iv = row['impliedVolatility']
                        c_vol = row['volume'] if not np.isnan(row['volume']) else 0
                        c_oi = row['openInterest'] if not np.isnan(row['openInterest']) else 0

                        # --- RENDER 7 SUB-TABS ---
                        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["1. Price Gap", "2. Volume", "3. IV", "4. Rule of 16", "5. Vol/OI", "6. Delta", "7. Max Pain"])
                        
                        # 1. GAP
                        with t1:
                            st.subheader("Price Gap")
                            gap = curr_p - prev_c
                            gp = (gap/prev_c)*100
                            st.metric("Gap", f"${gap:.2f}", f"{gp:.2f}%")
                            if abs(gap) < 0.5: st.success("✅ Small Gap.")
                            elif gap < -1: st.error("⚠️ Big Gap Down.")
                            else: st.info("ℹ️ Significant Gap.")

                        # 2. VOL
                        with t2:
                            st.subheader("Volume")
                            av = info.get('averageVolume', 0)
                            cv = info.get('volume', 0)
                            st.metric("Current Vol", f"{cv:,}", delta=f"{cv-av:,}")
                            st.write(f"Avg Vol: {av:,}")

                        # 3. IV
                        with t3:
                            st.subheader("IV Check")
                            ivp = c_iv * 100
                            st.metric("IV", f"{ivp:.2f}%")
                            if ivp < 20: st.write("🧊 Low IV")
                            elif ivp > 50: st.warning("🔥 High IV")
                            else: st.success("✅ Normal IV")

                        # 4. RULE 16
                        with t4:
                            st.subheader("Rule of 16")
                            dm_pct = ivp / 16
                            dm_dol = curr_p * (dm_pct/100)
                            c1, c2 = st.columns(2)
                            c1.metric("Exp Move $", f"${dm_dol:.2f}")
                            c2.metric("Target", f"${target_profit_move:.2f}")
                            if target_profit_move > dm_dol: st.error("❌ Target > Expected Move")
                            else: st.success("✅ Target Feasible")

                        # 5. VOL/OI
                        with t5:
                            st.subheader("Vol vs OI")
                            c1, c2 = st.columns(2)
                            c1.metric("Vol", f"{c_vol:,.0f}")
                            c2.metric("OI", f"{c_oi:,.0f}")
                            ratio = c_vol/c_oi if c_oi > 0 else 0
                            st.metric("Ratio", f"{ratio:.2f}x")
                            if c_vol > c_oi: st.markdown("### <span class='pass'>✅ PASS (Aggressive)</span>", unsafe_allow_html=True)
                            elif ratio > 0.5: st.markdown("### <span class='warning'>⚠️ WATCH</span>", unsafe_allow_html=True)
                            else: st.markdown("### <span class='fail'>❌ FAIL</span>", unsafe_allow_html=True)

                        # 6. DELTA
                        with t6:
                            st.subheader("Delta")
                            T_y = max((datetime.strptime(sel_date, '%Y-%m-%d').date() - date.today()).days / 365.0, 0.001)
                            dlt = calculate_delta(curr_p, strike_price, T_y, risk_free_rate, c_iv)
                            st.metric("Delta", f"{dlt:.2f}")
                            if dlt < 0.3: st.error("❌ High Risk")
                            elif dlt > 0.7: st.success("✅ Safe (Deep ITM)")
                            else: st.success("✅ Good Balance")

                        # 7. MAX PAIN
                        with t7:
                            st.subheader("Max Pain")
                            mp = calculate_max_pain(full_chain)
                            st.metric("Max Pain", f"${mp:.2f}", delta=f"{curr_p - mp:.2f}")
                            if abs(curr_p - mp) < 1: st.warning("🧲 Pinned")
                            else: st.info(f"Magnet Effect to ${mp:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
