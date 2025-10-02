# Stock Portfolio Momentum Forecaster

Python toolkit for building and training an LSTM model that forecasts short-term stock momentum (forward returns) from 30-day sequences of price, volume, and technical indicators. The resulting signals can be exported into Rotman Portfolio Manager or any other execution layer to drive allocation decisions.

## Features
- Automated price/volume ingestion via Yahoo Finance (yfinance).
- Feature engineering for SMA, EMA, MACD, RSI, volatility, and volume change.
- Chronological sequence builder for multi-ticker time-series windows.
- Configurable stacked LSTM model with early-stopping training loop.
- Persistence of the trained model and input scaler for downstream deployment.

## Getting Started
1. (Optional) Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model with default settings:
   ```bash
   python src/train.py \
       --tickers AAPL MSFT GOOGL AMZN \
       --start 2015-01-01 \
       --sequence-length 30 \
       --forecast-horizon 5
   ```
   Adjust tickers, window length, and horizon to match your Rotman Portfolio Manager strategy.

Training stores the model in `models/lstm_momentum.keras` and the feature scaler in `models/feature_scaler.pkl`. Metrics are printed to the console once training finishes.

## Using The Signals
- Load the saved model and scaler within your Rotman integration script to score the latest 30-day window for each asset.
- Convert the predicted momentum into position weights (e.g., via ranking, softmax, or threshold rules) before submitting trades to Rotman Portfolio Manager.
- Re-train periodically to incorporate the latest market regime changes.

## Notes
- Yahoo Finance requests require network access during training.
- TensorFlow can leverage GPU support if available; otherwise it defaults to CPU.
- Extend `data_pipeline.py` to add additional indicators (e.g., Bollinger Bands) or alternative forecast targets (e.g., classification of positive/negative momentum) as needed.
