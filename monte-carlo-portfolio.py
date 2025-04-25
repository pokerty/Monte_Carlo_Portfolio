import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Monte Carlo Portfolio Simulation")

# --- Title and Introduction ---
st.title("Monte Carlo Portfolio Simulation")
st.markdown("""
This dashboard performs a Monte Carlo simulation to analyze potential portfolio performance based on historical stock data.

1.  **Select Tickers & Date Range:** Choose the stocks for your portfolio and the historical period for analysis.
2.  **Review Initial Data:** Click 'Submit' to view historical prices, returns, performance metrics, and correlations.
3.  **Run Simulation:** Adjust the number of simulations and click 'Start Simulating!' to find the portfolio allocation with the highest historical Sharpe Ratio.
""")

st.divider()

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

st.sidebar.subheader("1. Select Tickers")
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NFLX', 'V', 'WMT', 'MCD', 'GE', 'GS', 'CVX', 'XOM']
selected_tickers = st.sidebar.multiselect("Tickers:", tickers, default=['AAPL', 'MSFT'], help="Select at least two tickers for the portfolio.")

st.sidebar.subheader("2. Select Date Range")
start_date = st.sidebar.date_input("Start Date:", datetime(2010, 1, 1).date())
end_date = st.sidebar.date_input("End Date:", datetime.now().date())

# --- Data Fetching and Preparation (Conditional on Tickers Selected) ---
if not selected_tickers or len(selected_tickers) < 2:
    st.warning("Please select at least two tickers in the sidebar.")
    st.stop() # Stop execution if not enough tickers are selected

@st.cache_data # Cache data fetching
def load_data(tickers, start, end):
    Close = pd.DataFrame()
    for ticker in tickers:
        try:
            Close[ticker] = yf.download(ticker, start=start, end=end, progress=False)['Close']
        except Exception as e:
            st.error(f"Failed to download data for {ticker}: {e}")
            return None, None # Return None if download fails
    if Close.empty or Close.isnull().all().all():
        st.error(f"Could not download any valid price data for the selected tickers and date range.")
        return None, None
    Return = Close.pct_change().dropna()
    return Close, Return

Close, Return = load_data(selected_tickers, start_date, end_date)

if Close is None or Return is None or Return.empty:
    st.error("Failed to load or process data. Please check tickers and date range.")
    st.stop()

# Calculate Risk-Free Rate (using cached data if possible)
@st.cache_data
def get_risk_free_rate(end_date):
    start_date_rf = end_date - timedelta(days=365)
    try:
        IRX = yf.download('^IRX', start=start_date_rf, end=end_date, progress=False)
        rf_series = IRX['Close'].pct_change().dropna()
        if rf_series.empty:
            return 0.0 # Default if no data
        else:
            return float(rf_series.mean())
    except Exception:
        return 0.0 # Default on error

rf = get_risk_free_rate(end_date)
st.sidebar.metric("Avg Daily Risk-Free Rate (Annualized ≈)", f"{rf:.6f}", f"{rf*252*100:.2f}%")

# --- Initial Data Display Section ---
st.header("Historical Data Overview")

if st.button("Show Historical Data", key="show_data_button"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Price History')
        st.line_chart(Close)

    with col2:
        st.subheader('Daily Returns')
        st.line_chart(Return)

    st.subheader('Performance Summary (Daily)')
    mean_sd_return = pd.DataFrame({'Mean Daily Return': Return.mean(), 'Std Dev': Return.std()}).T
    st.table(mean_sd_return.style.format("{:.4%}")) # Format as percentage

    st.subheader('Correlation Matrix')
    mat_cor = Return.corr()
    fig1, ax1 = plt.subplots(figsize=(8, 6)) # Adjust figure size
    sns.heatmap(mat_cor, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}, ax=ax1, linewidths=.5)
    plt.title('Correlation of Daily Returns', color='white')
    fig1.patch.set_alpha(0) # Make figure background transparent
    ax1.tick_params(colors='white', which='both')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig1)

st.divider()

# --- Monte Carlo Simulation Section ---
st.header("Monte Carlo Simulation")

st.sidebar.subheader("3. Simulation Settings")
n_simulation = st.sidebar.slider('Number of Simulations', 1000, 20000, 5000, step=1000, help="Higher numbers increase runtime but may find better results.")

if st.sidebar.button('Start Simulating!', key="start_simulation_button"):
    st.info(f"Running {n_simulation} simulations...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize variables to store results efficiently
    all_sharpe_ratios = []
    all_perform_metrics = {'return': [], 'std': []}
    max_sharpe_so_far = -np.inf
    best_weights = None
    best_perform_metrics = {'return': None, 'std': None}

    num_assets = Return.shape[1]

    for i in range(n_simulation):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Use dot product for faster calculation
        portfolio_return_series = Return.dot(weights)
        current_return = portfolio_return_series.mean()
        current_std = portfolio_return_series.std()

        sharpe = -np.inf
        if current_std != 0 and not np.isnan(current_std):
            sharpe = (current_return - rf) / current_std

        all_sharpe_ratios.append(sharpe)
        all_perform_metrics['return'].append(current_return)
        all_perform_metrics['std'].append(current_std)

        if sharpe > max_sharpe_so_far:
            max_sharpe_so_far = sharpe
            best_weights = weights
            best_perform_metrics['return'] = current_return
            best_perform_metrics['std'] = current_std

        progress = (i + 1) / n_simulation
        progress_bar.progress(progress)
        status_text.text(f"Simulation {i+1}/{n_simulation}")

    progress_bar.empty()
    status_text.success(f"Simulation Complete! Found best Sharpe Ratio: {max_sharpe_so_far:.4f}")

    # --- Display Simulation Results ---
    st.subheader("Simulation Results")

    col_res1, col_res2 = st.columns([2, 1]) # Make scatter plot wider

    with col_res1:
        st.write('Portfolio Return vs. Risk (Std Dev)')
        perform_df = pd.DataFrame(all_perform_metrics)
        # Add Sharpe Ratio to the DataFrame for coloring (optional but nice)
        perform_df['Sharpe Ratio'] = all_sharpe_ratios
        st.scatter_chart(perform_df, x='std', y='return', color='Sharpe Ratio', size=10) # Smaller points

    with col_res2:
        st.write('Sharpe Ratio Distribution')
        # Use numpy histogram for potentially better performance with large data
        hist_values, hist_bins = np.histogram(all_sharpe_ratios, bins=50)
        hist_df = pd.DataFrame({'Sharpe Ratio': hist_bins[:-1], 'Count': hist_values})
        st.bar_chart(hist_df.set_index('Sharpe Ratio'))
        st.metric("Highest Sharpe Ratio", f"{max_sharpe_so_far:.4f}")

    st.divider()

    # Display Optimal Portfolio Details
    if best_weights is not None:
        st.subheader('🏆 Optimal Portfolio (Highest Sharpe Ratio)')
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            st.write('Performance Metrics (Daily)')
            maxSharpe_perform_df = pd.DataFrame.from_dict(best_perform_metrics, orient='index', columns=['Value'])
            maxSharpe_perform_df.index.name = 'Metric'
            # Add Sharpe ratio to the performance table
            maxSharpe_perform_df.loc['Sharpe Ratio'] = max_sharpe_so_far
            st.table(maxSharpe_perform_df.style.format({'Value': '{:.4f}'}))

        with col_opt2:
            st.write('Optimal Asset Weights')
            maxSharpe_weights_df = pd.DataFrame({'Ticker': selected_tickers, 'Weight': best_weights})
            st.bar_chart(maxSharpe_weights_df.set_index('Ticker'))

    else:
        st.warning("No valid simulation results found.")

else:
    st.info("Configure settings in the sidebar and click 'Start Simulating!'")