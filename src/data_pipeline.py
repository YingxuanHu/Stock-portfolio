"""Data ingestion and feature engineering utilities for LSTM momentum forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class DataConfig:
    tickers: Iterable[str]
    start: str
    end: str | None
    sequence_length: int = 30
    forecast_horizon: int = 5


def download_price_data(tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data using yfinance with adjusted close prices."""
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by='ticker'
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data.stack(level=0, future_stack=True).rename_axis(index=['Date', 'Ticker']).sort_index()
    else:
        ticker = list(tickers)[0]
        data['Ticker'] = ticker
        data = data.reset_index().set_index(['Date', 'Ticker'])
    return data


def compute_technical_indicators(df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """Compute technical indicators (RSI, MACD, SMA, EMA, volatility) for each ticker."""

    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df = df.copy()
    grouped = []
    for ticker, sub_df in df.groupby(level='Ticker'):
        sub_df = sub_df.reset_index(level='Ticker', drop=True)
        close = sub_df['Close']
        volume = sub_df['Volume']
        feature_df = pd.DataFrame(index=sub_df.index)
        feature_df['close'] = close
        feature_df['volume'] = volume
        feature_df['return_1d'] = close.pct_change()
        feature_df['sma_10'] = close.rolling(10).mean()
        feature_df['sma_30'] = close.rolling(30).mean()
        feature_df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        feature_df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        feature_df['macd'] = feature_df['ema_12'] - feature_df['ema_26']
        feature_df['macd_signal'] = feature_df['macd'].ewm(span=9, adjust=False).mean()
        feature_df['macd_hist'] = feature_df['macd'] - feature_df['macd_signal']
        feature_df['rsi_14'] = rsi(close)
        feature_df['volatility_10'] = close.pct_change().rolling(10).std()
        feature_df['volume_change'] = volume.pct_change()
        feature_df['momentum_target'] = close.shift(-forecast_horizon) / close - 1
        feature_df['Ticker'] = ticker
        grouped.append(feature_df)

    full = pd.concat(grouped)
    full = full.dropna()
    return full


def build_sequences(
    feature_df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """Convert engineered features into 3D arrays for LSTM training."""
    if feature_columns is None:
        feature_columns = [
            'close', 'volume', 'return_1d', 'sma_10', 'sma_30', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist', 'rsi_14', 'volatility_10', 'volume_change'
        ]

    X_sequences: List[np.ndarray] = []
    y_values: List[float] = []
    tickers: List[str] = []
    end_dates: List[pd.Timestamp] = []

    for ticker, group in feature_df.groupby('Ticker'):
        group = group.copy()
        group.sort_index(inplace=True)
        values = group[feature_columns + ['momentum_target']]
        for idx in range(sequence_length, len(values) + 1):
            window = values.iloc[idx - sequence_length:idx]
            target = values['momentum_target'].iloc[idx - 1]
            X_sequences.append(window[feature_columns].to_numpy(dtype=np.float32))
            y_values.append(float(target))
            tickers.append(ticker)
            end_dates.append(window.index[-1])

    if not X_sequences:
        raise ValueError("Not enough observations to build sequences. Reduce sequence_length or extend the date range.")

    X = np.stack(X_sequences)
    y = np.array(y_values, dtype=np.float32)
    return X, y, tickers, pd.DatetimeIndex(end_dates)


def prepare_dataset(config: DataConfig) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    raw = download_price_data(config.tickers, config.start, config.end)
    features = compute_technical_indicators(raw, forecast_horizon=config.forecast_horizon)
    return build_sequences(
        feature_df=features,
        sequence_length=config.sequence_length,
    )
