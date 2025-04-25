# Monte Carlo Portfolio Simulation

This app was made by me to run Monte Carlo simulatios to analyze potential portfolio performance based on historical stock data, and visualize the results

## Features

*   Select multiple stock tickers.
*   Define a historical date range for analysis.
*   View historical price charts and daily returns.
*   Analyze performance metrics (mean return, standard deviation) and correlation matrices.
*   Run Monte Carlo simulations to find the portfolio allocation with the highest historical Sharpe Ratio.
*   Visualize simulation results (Return vs. Risk scatter plot, Sharpe Ratio distribution).
*   Display the optimal portfolio weights and performance metrics.

## Setup

1.  **Clone the repository (or download the files):**
    ```bash
    # If using git
    git clone <repository_url>
    cd Monte_Carlo_Ticker
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run monte-carlo-ticker.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  Use the sidebar to:
    *   Select the stock tickers for your portfolio.
    *   Choose the start and end dates for historical data.
4.  Click "Show Historical Data" to review the initial data analysis.
5.  Adjust the number of simulations in the sidebar.
6.  Click "Start Simulating!" to run the Monte Carlo simulation and find the optimal portfolio allocation.
