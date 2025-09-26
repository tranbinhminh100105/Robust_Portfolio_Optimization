import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === CONFIGURATION ===
use_custom_clusters = True

custom_clusters = {
    1: ["AMZN", "V"],
    2: ["AAPL", "NVDA", "META", "UNH"],
    3: ["MSFT", "GOOGL", "TSLA", "JPM"]

    #4: ["AMZN", "V", "AAPL", "NVDA", "META", "UNH", "MSFT", "GOOGL", "TSLA", "JPM"]

    # 5: ["AMZN"],
    # 6: ["V"],
    # 7: ["AAPL"],
    # 8: ["NVDA"],
    # 9: ["META"],
    # 10: ["UNH"],
    # 11: ["MSFT"],
    # 12: ["GOOGL"],
    # 13: ["TSLA"],
    # 14: ["JPM"]


}

train_start = "2010-01-01"
train_end = "2019-12-31"
test_start = "2020-01-01"
test_end = "2024-01-01"

# === LOAD AND PREPARE DATA ===
df_raw = pd.read_csv("combined_10_stocks.csv", parse_dates=["Date"])
df = df_raw.pivot(index="Date", columns="Ticker", values="Adj Close")

# Train/test split
df_train = df.loc[train_start:train_end].dropna()
df_test = df.loc[test_start:test_end].dropna()

log_returns_train = np.log(df_train / df_train.shift(1)).dropna()
log_returns_test = np.log(df_test / df_test.shift(1)).dropna()

# === CLUSTERING ===
if use_custom_clusters:
    cluster_map = {}
    for cluster_id, tickers in custom_clusters.items():
        for ticker in tickers:
            cluster_map[ticker] = cluster_id
    cluster_map = pd.Series(cluster_map)
else:
    X = np.stack([log_returns_train.mean(), log_returns_train.std()], axis=1)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_map = pd.Series(labels, index=log_returns_train.columns)


# === OPTIMIZATION FUNCTION ===
def portfolio_optimization(mean_returns, cov_matrix, risk_free_rate=0):
    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))

    def sharpe_ratio(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_volatility

    result = minimize(sharpe_ratio, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x

# === DESCRIPTIVE STATISTICS TABLE (TRAINING SET) ===
print("\n📊 Table 1: Descriptive Statistics of Log Returns (Train Set)")
print(f"{'Ticker':<10}{'Mean':<10}{'Std':<10}")

log_stats = log_returns_train.describe().T[["mean", "std"]]

for ticker, row in log_stats.iterrows():
    print(f"{ticker:<10}{row['mean']:.4f}   {row['std']:.4f}")

# === OPTIMIZE PER CLUSTER ON TRAINING DATA ===
optimized_weights_per_cluster = {}

for cluster_id in sorted(cluster_map.unique()):
    cluster_tickers = cluster_map[cluster_map == cluster_id].index
    cluster_returns = log_returns_train[cluster_tickers]

    mean_returns = cluster_returns.mean().values
    cov_matrix = cluster_returns.cov().values

    weights = portfolio_optimization(mean_returns, cov_matrix)
    optimized_weights_per_cluster[cluster_id] = dict(zip(cluster_tickers, weights))

# === DISPLAY OPTIMIZED WEIGHTS ===
for cluster_id, weights in optimized_weights_per_cluster.items():
    print(f"\n📊 Cluster {cluster_id} Optimized Portfolio Weights:")
    for stock, weight in weights.items():
        print(f"  {stock}: {weight:.4f}")

# === BACKTEST WITH METRICS TABLE ===
print("\n📈 Backtesting Performance Summary (2020–2024):")

# Table header
print(f"{'Cluster':<10}{'Total Return (%)':<20}{'Annual Return (%)':<22}{'Volatility (%)':<18}{'Sharpe Ratio':<15}")

trading_days = 252
risk_free_rate = 0

for cluster_id, weights_dict in optimized_weights_per_cluster.items():
    tickers = list(weights_dict.keys())
    weights = np.array(list(weights_dict.values()))

    test_returns = log_returns_test[tickers].dropna()
    daily_returns = test_returns.dot(weights)

    # Calculate metrics
    cumulative_return = np.exp(daily_returns.cumsum()) - 1
    total_return = cumulative_return.iloc[-1]

    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    annualized_return = avg_daily_return * trading_days
    annualized_volatility = std_daily_return * np.sqrt(trading_days)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    print(f"{cluster_id:<10}{total_return * 100:<20.2f}{annualized_return * 100:<22.2f}"
          f"{annualized_volatility * 100:<18.2f}{sharpe_ratio:<15.2f}")

    # Optional: Plot
    plt.plot(cumulative_return * 100, label=f"Cluster {cluster_id}")


# === PLOT RESULTS ===
plt.title("Cumulative Returns (Test Period)")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()