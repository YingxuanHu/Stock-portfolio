"""Model utilities for the LSTM momentum forecaster."""
from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape: tuple[int, int], learning_rate: float = 1e-3) -> Sequential:
    """Construct a stacked LSTM regression model."""
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, name='momentum_output'),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model
