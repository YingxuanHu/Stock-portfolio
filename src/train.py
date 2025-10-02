"""CLI to train an LSTM model that forecasts stock momentum for portfolio allocation."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_pipeline import DataConfig, prepare_dataset
from modeling import build_lstm_model


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(dates)
    X = X[order]
    y = y[order]
    dates = dates[order]

    split_idx = int(len(X) * train_ratio)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Train ratio produces an empty train or test split")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def scale_sequences(scaler: StandardScaler, X: np.ndarray, fit: bool = False) -> np.ndarray:
    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    if fit:
        scaler.fit(X_flat)
    transformed = scaler.transform(X_flat)
    return transformed.reshape(n_samples, seq_len, n_features)


def train_model(
    tickers: list[str],
    start: str,
    end: str | None,
    sequence_length: int,
    forecast_horizon: int,
    train_ratio: float,
    epochs: int,
    batch_size: int,
    model_path: Path,
    scaler_path: Path,
) -> dict:
    config = DataConfig(
        tickers=tickers,
        start=start,
        end=end,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )
    X, y, _, dates = prepare_dataset(config)
    X_train, X_test, y_train, y_test, dates_train, dates_test = chronological_split(
        X=X,
        y=y,
        dates=dates.to_numpy(),
        train_ratio=train_ratio,
    )

    scaler = StandardScaler()
    X_train_scaled = scale_sequences(scaler, X_train, fit=True)
    X_test_scaled = scale_sequences(scaler, X_test)

    model = build_lstm_model(input_shape=X_train_scaled.shape[1:])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
    ]

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    test_predictions = model.predict(X_test_scaled).reshape(-1)
    mse = mean_squared_error(y_test, test_predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, test_predictions)
    direction_accuracy = float(np.mean(np.sign(test_predictions) == np.sign(y_test)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    with scaler_path.open('wb') as f:
        pickle.dump(scaler, f)

    metrics = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'rmse': float(rmse),
        'mae': float(mae),
        'directional_accuracy': direction_accuracy,
        'epochs_trained': len(history.history['loss']),
        'train_end_date': str(dates_train[-1]),
        'test_start_date': str(dates_test[0]),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM to forecast stock momentum across tickers.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN'], help='List of tickers to include.')
    parser.add_argument('--start', default='2015-01-01', help='Historical start date (YYYY-MM-DD).')
    parser.add_argument('--end', default=None, help='Historical end date (YYYY-MM-DD).')
    parser.add_argument('--sequence-length', type=int, default=30, help='Number of timesteps in each training window.')
    parser.add_argument('--forecast-horizon', type=int, default=5, help='Days ahead to forecast momentum.')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Fraction of samples used for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs for the LSTM model.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for model training.')
    parser.add_argument('--model-path', default='models/lstm_momentum.keras', help='Where to save the trained model.')
    parser.add_argument('--scaler-path', default='models/feature_scaler.pkl', help='Where to persist the feature scaler.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=Path(args.model_path),
        scaler_path=Path(args.scaler_path),
    )
    rmse_pct = metrics['rmse'] * 100
    mae_pct = metrics['mae'] * 100

    print("Training completed.")
    print(" ─ Model fit")
    print(f"   • Train samples: {metrics['train_samples']:,}")
    print(f"   • Test samples:  {metrics['test_samples']:,}")
    print(f"   • Epochs:        {metrics['epochs_trained']}")

    print(" ─ Test evaluation (forecast horizon)")
    print(f"   • RMSE: {metrics['rmse']:.6f} ({rmse_pct:.2f}%)")
    print(f"   • MAE:  {metrics['mae']:.6f} ({mae_pct:.2f}%)")
    print(f"   • Directional accuracy: {metrics['directional_accuracy']:.3f} ({metrics['directional_accuracy'] * 100:.1f}%)")

    print(" ─ Chronology")
    print(f"   • Train end: {metrics['train_end_date']}")
    print(f"   • Test start: {metrics['test_start_date']}")


if __name__ == '__main__':
    main()
