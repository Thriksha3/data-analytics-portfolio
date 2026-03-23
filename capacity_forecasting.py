"""
Time Series Forecasting — Demand & Capacity Analysis
======================================================
Analytical forecasting pipeline for operational planning:
- Trend decomposition & seasonality analysis
- Stationarity testing (ADF test)
- ARIMA / SARIMAX modeling
- Rolling statistics & anomaly detection
- 90-day forward projections with confidence intervals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def generate_capacity_data(days: int = 1095) -> pd.DataFrame:
    """Generate 3 years of daily infrastructure capacity data with realistic patterns."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")

    # Base trend (gradual increase)
    trend = np.linspace(60, 78, days)

    # Weekly seasonality (lower on weekends)
    weekly = -5 * np.sin(2 * np.pi * np.arange(days) / 7)

    # Monthly seasonality (end-of-month spikes)
    monthly = 3 * np.sin(2 * np.pi * np.arange(days) / 30.5)

    # Yearly seasonality (Q4 higher)
    yearly = 4 * np.sin(2 * np.pi * (np.arange(days) - 90) / 365)

    # Noise + occasional anomalies
    noise = np.random.normal(0, 2, days)
    anomalies = np.zeros(days)
    anomaly_idx = np.random.choice(days, 15, replace=False)
    anomalies[anomaly_idx] = np.random.uniform(10, 20, 15)

    capacity_pct = trend + weekly + monthly + yearly + noise + anomalies
    capacity_pct = np.clip(capacity_pct, 20, 100)

    df = pd.DataFrame({
        "date": dates,
        "capacity_utilization_pct": capacity_pct.round(2),
        "requests_count": (capacity_pct * 1000 + np.random.normal(0, 500, days)).clip(0).astype(int),
        "error_rate_pct": (np.random.exponential(0.5, days) + capacity_pct / 100).round(3),
    })
    df = df.set_index("date")
    return df


def analyze_trend_and_seasonality(series: pd.Series, period: int = 7):
    """Decompose time series into trend, seasonal, and residual components."""
    print(f"\n{'='*50}")
    print(f"  DECOMPOSITION ANALYSIS")
    print(f"{'='*50}")

    decomposition = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed", color="#2C3E50")
    decomposition.trend.plot(ax=axes[1], title="Trend", color="#E74C3C")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonality", color="#3498DB")
    decomposition.resid.plot(ax=axes[3], title="Residual", color="#95A5A6")

    for ax in axes:
        ax.set_ylabel("")
    plt.suptitle("Time Series Decomposition", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("output/decomposition.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"  ✓ Decomposition saved to output/")
    return decomposition


def test_stationarity(series: pd.Series, window: int = 30):
    """Augmented Dickey-Fuller test for stationarity with rolling statistics."""
    result = adfuller(series.dropna(), autolag="AIC")

    print(f"\n{'─'*50}")
    print(f"  STATIONARITY TEST (ADF)")
    print(f"{'─'*50}")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.6f}")
    print(f"  Lags Used:      {result[2]}")
    for key, val in result[4].items():
        print(f"  Critical ({key}): {val:.4f}")
    stationary = result[1] < 0.05
    print(f"  Stationary:     {'YES ✓' if stationary else 'NO ✗ — differencing needed'}")
    print(f"{'─'*50}")

    # Rolling statistics plot
    fig, ax = plt.subplots(figsize=(14, 5))
    series.plot(ax=ax, alpha=0.4, label="Original", color="#95A5A6")
    series.rolling(window=window).mean().plot(ax=ax, label=f"Rolling Mean ({window}d)",
                                                color="#E74C3C", linewidth=2)
    series.rolling(window=window).std().plot(ax=ax, label=f"Rolling Std ({window}d)",
                                               color="#3498DB", linewidth=2)
    ax.set_title("Rolling Statistics", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/stationarity.png", dpi=150, bbox_inches="tight")
    plt.show()

    return stationary, result


def detect_anomalies(series: pd.Series, window: int = 30, sigma: float = 2.5) -> pd.DataFrame:
    """Detect anomalies using rolling mean ± sigma standard deviations."""
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()

    upper = rolling_mean + sigma * rolling_std
    lower = rolling_mean - sigma * rolling_std

    anomalies = series[(series > upper) | (series < lower)]

    fig, ax = plt.subplots(figsize=(14, 5))
    series.plot(ax=ax, alpha=0.5, label="Observed", color="#2C3E50")
    rolling_mean.plot(ax=ax, label="Rolling Mean", color="#3498DB", linewidth=2)
    ax.fill_between(series.index, lower, upper, alpha=0.15, color="#3498DB", label=f"±{sigma}σ Band")
    ax.scatter(anomalies.index, anomalies.values, color="#E74C3C", s=40, zorder=5, label=f"Anomalies ({len(anomalies)})")
    ax.set_title("Anomaly Detection", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/anomalies.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"  ✓ Detected {len(anomalies)} anomalies ({len(anomalies)/len(series)*100:.1f}%)")
    return anomalies


def forecast_arima(series: pd.Series, forecast_days: int = 90,
                   order: tuple = (2, 1, 2)) -> pd.DataFrame:
    """
    Fit ARIMA model and produce forward projections with confidence intervals.
    """
    # Train/test split (last 30 days as holdout)
    train = series[:-30]
    test = series[-30:]

    # Fit model
    model = ARIMA(train, order=order)
    fitted = model.fit()

    # Evaluate on holdout
    test_forecast = fitted.forecast(steps=30)
    mae = mean_absolute_error(test, test_forecast)
    rmse = np.sqrt(mean_squared_error(test, test_forecast))

    print(f"\n{'='*50}")
    print(f"  ARIMA{order} FORECAST")
    print(f"{'='*50}")
    print(f"  Holdout MAE:  {mae:.2f}")
    print(f"  Holdout RMSE: {rmse:.2f}")
    print(f"  AIC:          {fitted.aic:.2f}")

    # Refit on full data and forecast forward
    full_model = ARIMA(series, order=order)
    full_fitted = full_model.fit()
    forecast_result = full_fitted.get_forecast(steps=forecast_days)
    forecast_mean = forecast_result.predicted_mean
    confidence = forecast_result.conf_int(alpha=0.05)

    # Create forecast dataframe
    forecast_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                                    periods=forecast_days, freq="D")
    forecast_df = pd.DataFrame({
        "forecast": forecast_mean.values,
        "lower_95": confidence.iloc[:, 0].values,
        "upper_95": confidence.iloc[:, 1].values,
    }, index=forecast_dates)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    series[-180:].plot(ax=ax, label="Historical (last 6 months)", color="#2C3E50", linewidth=1.5)
    forecast_df["forecast"].plot(ax=ax, label=f"{forecast_days}-Day Forecast", color="#E74C3C",
                                  linewidth=2, linestyle="--")
    ax.fill_between(forecast_df.index, forecast_df["lower_95"], forecast_df["upper_95"],
                    alpha=0.2, color="#E74C3C", label="95% Confidence")
    ax.set_title(f"ARIMA{order} — {forecast_days}-Day Forward Projection", fontsize=13, fontweight="bold")
    ax.set_ylabel("Capacity Utilization (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/forecast.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"  Forecast range: {forecast_df['forecast'].min():.1f}% — {forecast_df['forecast'].max():.1f}%")
    print(f"  ✓ Forecast saved to output/")

    return forecast_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("Generating 3 years of capacity data...")
    df = generate_capacity_data(1095)
    series = df["capacity_utilization_pct"]

    print(f"Dataset: {len(df):,} days ({df.index.min().date()} to {df.index.max().date()})")

    # 1. Decomposition
    decomp = analyze_trend_and_seasonality(series, period=7)

    # 2. Stationarity test
    is_stationary, adf_result = test_stationarity(series)

    # 3. Anomaly detection
    anomalies = detect_anomalies(series, window=30, sigma=2.5)

    # 4. Forecast
    forecast = forecast_arima(series, forecast_days=90, order=(2, 1, 2))

    print("\n✓ All analyses complete. Check output/ folder.")
