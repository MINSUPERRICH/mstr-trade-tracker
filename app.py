import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Option Strategy Simulator",
    page_icon="📈",
    layout="wide"
)

# --- BLACK-SCHOLES FORMULA (The Math Engine) ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: Underlying Stock Price
    K: Strike Price
    T: Time to Expiration (years)
    r: Risk-free Interest Rate (decimal, e.g., 0.04)
    sigma: Implied Volatility (decimal, e.g., 0.80)
    """
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# --- SIDEBAR: INPUT YOUR TRADE ---
st.sidebar.header("📝 1. Trade Details (Input)")

symbol = st.sidebar.text_input("Stock Symbol", value="MSTR").upper()
purchase_date = st.sidebar.date_input("Purchase Date", value=date(2024, 12, 24))
entry_price = st.sidebar.number_input("Entry Price (Premium Paid)", value=8.55, step=0.10)
contracts = st.sidebar.number_input("Number of Contracts", value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Contract Specs")
current_stock_price = st.sidebar.number_input("Current Stock Price ($)", value=158.00, step=0.50)
strike_price = st.sidebar.number_input("Strike Price ($)", value=157.50, step=0.50)
expiration_date = st.sidebar.date_input("Expiration Date", value=date(2026, 1, 9))
implied_volatility = st.sidebar.slider("Implied Volatility (IV %)", 10, 200, 95, help="Higher IV = More Expensive Options. Check your broker for this number.") / 100.0
risk_free_rate = 0.045 # Approx 4.5% interest rate

# --- MAIN PAGE: THE SIMULATOR ---
st.title(f"🔮 {symbol} Option Profit Simulator")
st.markdown("Use the sliders below to ask: **'What happens if I sell on X date at Y price?'**")

# --- SIMULATION SLIDERS ---
st.markdown("### 🎛️ Simulation Controls")
col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    sim_date = st.slider(
        "📅 Date to Sell (Simulated)", 
        min_value=purchase_date, 
        max_value=expiration_date, 
        value=date.today() + timedelta(days=5),
        format="MMM DD, YYYY"
    )

with col_sim2:
    sim_price = st.slider(
        "💲 Stock Price at Sale (Simulated)", 
        min_value=float(current_stock_price * 0.5), 
        max_value=float(current_stock_price * 2.0), 
        value=float(current_stock_price),
        step=1.0
    )

# --- CALCULATE SIMULATION MATH ---
days_to_expiry_sim = (expiration_date - sim_date).days
time_to_expiry_years = max(days_to_expiry_sim / 365.0, 0.0001)

# Calculate Projected Option Price using Black-Scholes
projected_option_price = black_scholes(sim_price, strike_price, time_to_expiry_years, risk_free_rate, implied_volatility)

# Calculate P/L
total_cost = entry_price * 100 * contracts
exit_value = projected_option_price * 100 * contracts
net_profit = exit_value - total_cost
roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

# --- DISPLAY RESULTS ---
st.markdown("---")
st.subheader("📊 Projected Results")

res_col1, res_col2, res_col3, res_col4 = st.columns(4)

with res_col1:
    st.metric("📆 Date", f"{sim_date.strftime('%b %d')}", f"{(sim_date - date.today()).days} days away")

with res_col2:
    st.metric("💲 Target Stock Price", f"${sim_price:.2f}")

with res_col3:
    st.metric("📈 Est. Option Price", f"${projected_option_price:.2f}", delta=f"{projected_option_price - entry_price:.2f}")

with res_col4:
    color = "normal" if net_profit >= 0 else "inverse"
    st.metric("💰 Net Profit/Loss", f"${net_profit:,.2f}", delta=f"{roi:.1f}%", delta_color=color)

# --- HEATMAP VISUALIZATION ---
st.markdown("### 🗺️ Profit/Loss Heatmap")
st.write("This map shows your profit (Green) or loss (Red) at different prices and dates.")

# Generate Data for Heatmap
prices = np.linspace(current_stock_price * 0.8, current_stock_price * 1.5, 20)
future_dates = [date.today() + timedelta(days=x) for x in range(0, 60, 5)] # Next 60 days
heatmap_data = []

for d in future_dates:
    t_years = max((expiration_date - d).days / 365.0, 0.0001)
    for p in prices:
        opt_price = black_scholes(p, strike_price, t_years, risk_free_rate, implied_volatility)
        pl = (opt_price - entry_price) * 100 * contracts
        heatmap_data.append({
            "Date": d.strftime('%Y-%m-%d'),
            "Stock Price": round(p, 2),
            "Profit": round(pl, 2)
        })

df_heatmap = pd.DataFrame(heatmap_data)

# Create Chart
heatmap = alt.Chart(df_heatmap).mark_rect().encode(
    x='Date:O',
    y='Stock Price:O',
    color=alt.Color('Profit', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
    tooltip=['Date', 'Stock Price', 'Profit']
).properties(height=400)

st.altair_chart(heatmap, use_container_width=True)

# --- EXPLANATION SECTION ---
with st.expander("ℹ️ How does this math work? (The 'Greeks')"):
    st.markdown("""
    This simulator uses the **Black-Scholes Model**, which is the standard for option pricing.
    
    * **Delta (Price):** If the stock moves up $1, your option moves up by Delta (approx 0.50 to 1.00).
    * **Theta (Time):** Every day that passes, the option loses value. This calculator accounts for that "decay."
    * **Vega (Volatility):** We assumed Volatility stays constant. If the market panics, Volatility goes up, and your option might be worth *more* than shown here.
    """)
