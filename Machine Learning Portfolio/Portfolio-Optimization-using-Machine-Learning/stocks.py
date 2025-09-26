import yfinance as yf
import pandas as pd

# List of 10 stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'UNH']

# Download all in one go
data = yf.download(tickers, start="2010-01-01", end="2024-01-01", auto_adjust=False, group_by='ticker')

# Create a combined DataFrame with a flat structure
combined_data = pd.DataFrame()

for ticker in tickers:
    stock_df = data[ticker].copy()
    stock_df['Ticker'] = ticker
    stock_df.reset_index(inplace=True)
    combined_data = pd.concat([combined_data, stock_df], ignore_index=True)

# Save to CSV
combined_data.to_csv('combined_10_stocks.csv', index=False)

print("✅ Downloaded and saved to combined_10_stocks.csv")
