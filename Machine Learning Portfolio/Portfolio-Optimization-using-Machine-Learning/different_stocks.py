import yfinance as yf
import pandas as pd

# --- Define stock tickers by category ---
tech_stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'ADBE', 'CRM', 'INTC'
]

bank_stocks = [
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'PNC', 'USB', 'TD', 'BK'
]

insurance_stocks = [
    'AIG', 'ALL', 'TRV', 'MET', 'PRU', 'HIG', 'LNC', 'PGR', 'CINF', 'AFG'
]

trade_stocks = [
    'WMT', 'COST', 'HD', 'TGT', 'LOW', 'DG', 'DLTR', 'KR', 'BJ', 'AMZN'
]

other_stocks = [
    'UNH', 'PG', 'KO', 'PEP', 'MCD', 'DIS', 'NKE', 'VZ', 'T', 'CSCO'
]

# Combine all tickers for one download
all_tickers = tech_stocks + bank_stocks + insurance_stocks + trade_stocks + other_stocks

# Download data for all tickers
data = yf.download(all_tickers, start="2010-01-01", end="2024-01-01", auto_adjust=False, group_by='ticker')

# Function to process and save category
def save_category(tickers, category_name):
    try:
        combined_data = pd.DataFrame()
        for ticker in tickers:
            stock_df = data[ticker].copy()
            stock_df['Ticker'] = ticker
            stock_df.reset_index(inplace=True)
            combined_data = pd.concat([combined_data, stock_df], ignore_index=True)
        file_name = f"{category_name}_stocks.csv"
        combined_data.to_csv(file_name, index=False)
        print(f"✅ Saved {category_name} stocks to {file_name}")
    except Exception as e:
        print(f'Error: {e}')

# Save each category separately
save_category(tech_stocks, "tech")
save_category(bank_stocks, "bank")
save_category(insurance_stocks, "insurance")
save_category(trade_stocks, "trade")
save_category(other_stocks, "other")
