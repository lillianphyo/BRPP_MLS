import os
import numpy as np
import tensorflow as tf
import logging
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from .preprocessing import log_transform, create_sequences

# Add the parent directory to the system path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Configure logging with datetime
logging.basicConfig(filename=os.path.join(config.log_dir, 'training.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(df_log, seq_length=12, epochs=100, batch_size=64):
    # Prepare data for LSTM
    data = df_log[['o_rice', 'h_rice', 'l_rice', 'c_rice']].values
    X, y = create_sequences(data, seq_length)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Build the LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

        # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    return model, history, X_test, y_test

def save_model(model, model_name):
    # model_path = os.path.join(config.model_dir, model_name)
    # model.save(model_path)
    # logging.info(f"Model saved as {model_path}")
    model_dir = config.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    ############################################
    # tf.saved_model.save(model, model_dir)
    ############################################
    print(f"Model saved as {model_path}")

    # Log model summary
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_summary_file = os.path.join(config.log_dir, 'model_summary_{current_datetime}.txt')
    with open(model_summary_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logging.info(f"Model summary saved as {model_summary_file}")
