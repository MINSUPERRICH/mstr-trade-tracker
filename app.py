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
#  MATH ENGINE
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
    """Calculates the Delta of the option."""
    if T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

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

# --- SIDEBAR ---
st.sidebar.header("🌍 Market Conditions")
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

# --- TABS ---
tab_math, tab_dashboard, tab_ai, tab_catalyst = st.tabs(["⚔️ Strategy Battle", "📊 Market Dashboard", "📸 Chart Analyst", "📅 Catalyst & Checklist"])

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
#  TAB 4: CATALYST & CHECKLIST (UPDATED!)
# =========================================================
with tab_catalyst:
    st.subheader("📅 Catalyst & Pre-Trade Checklist")
    
    cat_sym = st.text_input("Enter Symbol to Check:", value=symbol, key="cat_sym")
    
    # 1. CATALYSTS
    if st.button("🔎 Check Catalysts"):
        tick = yf.Ticker(cat_sym)
        st.write("---")
        
        # Earnings
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

        # News
        st.markdown("---")
        st.markdown("### 2. Recent News")
        try:
            news = tick.news
            if news:
                for n in news[:3]:
                    with st.expander(f"📰 {n['title']}"): st.write(f"Source: {n['publisher']} | [Link]({n['link']})")
            else: st.info("No news found.")
        except: st.info("News feed unavailable.")

    # 2. THE 6-POINT CHECKLIST
    st.markdown("---")
    st.subheader("✅ The 6-Point Trade Checklist")
    st.write("Evaluate your setup before entry. Data fetches live.")
    
    if st.button("🚀 Run Checklist"):
        tick = yf.Ticker(cat_sym)
        checklist_data = []
        
        # Fetch Data
        try:
            hist = tick.history(period="1mo")
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            today_open = hist['Open'].iloc[-1]
            avg_vol = hist['Volume'].mean()
            today_vol = hist['Volume'].iloc[-1]
            
            # 1. Price Gap Check
            gap = abs(today_open - prev_close)
            status_gap = "⚠️ Careful" if gap > 1.00 else "✅ Pass"
            checklist_data.append({"Check": "1. Price Gap", "Value": f"${gap:.2f}", "Result": status_gap, "Note": "Is gap > $1.00?"})
            
            # 2. Volume Activity Check
            status_vol = "✅ Pass" if today_vol > avg_vol else "⚠️ Low"
            checklist_data.append({"Check": "2. Volume Activity", "Value": f"{today_vol/1000000:.1f}M", "Result": status_vol, "Note": "Is Vol > Avg?"})
            
            # 3. IV Fear Check (Simple Trend)
            # Fetch Option Chain for IV
            try:
                # Find closest date
                avail_dates = tick.options
                if avail_dates:
                    target_dt = datetime.strptime(expiration_date.strftime('%Y-%m-%d'), '%Y-%m-%d').date()
                    closest_date = min(avail_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - target_dt))
                    chain = tick.option_chain(closest_date).calls
                    contract = chain[chain['strike'] == strike_price]
                    
                    if not contract.empty:
                        iv = contract.iloc[0]['impliedVolatility']
                        opt_vol = contract.iloc[0]['volume']
                        opt_oi = contract.iloc[0]['openInterest']
                        
                        # Rule of 16 (Daily Move)
                        daily_move = current_price * (iv / 16)
                        checklist_data.append({"Check": "3. IV Check", "Value": f"{iv*100:.1f}%", "Result": "ℹ️ Info", "Note": "Check IV Rank manually if needed."})
                        
                        # 4. Rule of 16 Reality Check
                        checklist_data.append({"Check": "4. Rule of 16 (Exp. Move)", "Value": f"${daily_move:.2f}", "Result": "ℹ️ Info", "Note": "Is Target < This?"})
                        
                        # 5. Vol vs OI
                        status_voi = "✅ Pass" if opt_vol > opt_oi else "⚠️ Low Vol"
                        checklist_data.append({"Check": "5. Vol vs OI", "Value": f"{opt_vol} / {opt_oi}", "Result": status_voi, "Note": "Is Vol > OI?"})
                        
                        # 6. Delta Check
                        # Calculate Delta
                        t_years = max((datetime.strptime(closest_date, '%Y-%m-%d').date() - date.today()).days / 365.0, 0.0001)
                        delta = calculate_delta(current_price, strike_price, t_years, risk_free_rate, iv)
                        status_delta = "✅ Pass" if delta >= 0.30 else "⚠️ Low Delta"
                        checklist_data.append({"Check": "6. Delta Check", "Value": f"{delta:.2f}", "Result": status_delta, "Note": "Is Delta > 0.30?"})
                        
                    else:
                        st.error("Strike not found for checklist.")
            except Exception as e:
                checklist_data.append({"Check": "Option Data", "Value": "Error", "Result": "❌ Fail", "Note": str(e)})

            # Display Table
            df_check = pd.DataFrame(checklist_data)
            def highlight_res(val):
                if "Pass" in val: return 'color: green; font-weight: bold'
                if "Careful" in val or "Low" in val: return 'color: orange; font-weight: bold'
                return ''
            st.dataframe(df_check.style.applymap(highlight_res, subset=['Result']), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error running checklist: {e}")
