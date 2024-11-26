import numpy as np
import pandas as pd

# Apply log transformation
def log_transform(df, cols):
    df_log = df.copy()
    for col in cols:
        df_log[col] = np.log1p(df[col])  # Using log1p for numerical stability (log(1+x))
    return df_log

# Inverse log transformation
def inverse_log_transform(df_log, cols):
    df_original = df_log.copy()
    for col in cols:
        df_original[col] = np.expm1(df_log[col])  # Using expm1 to reverse the log1p transformation
    return df_original

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # All columns except target
        y.append(data[i+seq_length, -1])  # Only the target column
    return np.array(X), np.array(y)
