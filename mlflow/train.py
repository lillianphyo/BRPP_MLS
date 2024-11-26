# import os
# import numpy as np
# import tensorflow as tf
# import logging
# import datetime
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
# from sklearn.model_selection import train_test_split
# from preprocessing import log_transform, create_sequences

# # Add the parent directory to the system path
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import config

# # Configure logging with datetime
# logging.basicConfig(filename=os.path.join(config.log_dir, 'training.log'),
#                     level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# def train_model(df_log, seq_length=12, epochs=100, batch_size=64):
#     # Prepare data for LSTM
#     data = df_log[['o_rice', 'h_rice', 'l_rice', 'c_rice']].values
#     X, y = create_sequences(data, seq_length)

#     # Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     # Build the Bidirectional LSTM model
#     model = Sequential()
#     model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(Dropout(0.3))
#     model.add(LSTM(units=100, return_sequences=False))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=1))

#     # Compile the model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

#     # Train the model
#     history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

#     return model, history, X_test, y_test

# def save_model(model, model_name):
#     # model_path = os.path.join(config.model_dir, model_name)
#     # model.save(model_path)
#     # logging.info(f"Model saved as {model_path}")
#     model_dir = config.model_dir
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, model_name)
#     model.save(model_path)
#     ############################################
#     # tf.saved_model.save(model, model_dir)
#     ############################################
#     print(f"Model saved as {model_path}")

#     # Log model summary
#     current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     model_summary_file = os.path.join(config.log_dir, 'model_summary_{current_datetime}.txt')
#     with open(model_summary_file, 'w') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#     logging.info(f"Model summary saved as {model_summary_file}")

#mlflow
# import os
# import numpy as np
# import tensorflow as tf
# import logging
# import datetime
# import mlflow
# import mlflow.keras
# import psutil

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from preprocessing import create_sequences

# import config

# # Configure logging with datetime
# logging.basicConfig(
#     filename=os.path.join(config.log_dir, 'training.log'),
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# def train_model(df_log, market, seq_length=12, epochs=100, batch_size=64):
#     # Prepare data for LSTM
#     data = df_log[['o_rice', 'h_rice', 'l_rice', 'c_rice']].values
#     X, y = create_sequences(data, seq_length)

#     # Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     # Build the LSTM model
#     model = Sequential()
#     model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(Dropout(0.3))
#     model.add(LSTM(units=100, return_sequences=False))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=1))

#     # Compile the model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

#     # Start MLflow tracking
#     mlflow.set_tracking_uri(config.mlflow_tracking_uri)
#     with mlflow.start_run():
#         # Log parameters
#         mlflow.log_param("market", market)
#         mlflow.log_param("epochs", epochs)
#         mlflow.log_param("batch_size", batch_size)

#         # Track system metrics
#         mlflow.log_metric("cpu_percent", psutil.cpu_percent(interval=None))
#         mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)
        
#         # Train the model
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

#         # Save the model
#         model_name = f"lstm_model_{market}.keras"
#         save_model(model, model_name)
        
#         # Log model
#         mlflow.keras.log_model(model, "model")
        
#         # Predict on the test set
#         y_pred = model.predict(X_test)

#         # Calculate and log metrics
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         mse = mean_squared_error(y_test, y_pred)

#         mlflow.log_metric("MAE", mae)
#         mlflow.log_metric("RMSE", rmse)
#         mlflow.log_metric("MSE", mse)
        
#         logging.info(f"Market: {market}, MAE: {mae}, RMSE: {rmse}, MSE: {mse}")

#     return model, history, X_test, y_test

# def save_model(model, model_name):
#     model_dir = config.model_dir
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, model_name)
#     model.save(model_path)
#     logging.info(f"Model saved as {model_path}")

#     # Log model summary
#     current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     model_summary_file = os.path.join(config.log_dir, f'model_summary_{current_datetime}.txt')
#     with open(model_summary_file, 'w') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#     logging.info(f"Model summary saved as {model_summary_file}")

#mlflow plus precision, recall and F1 score
# import os
# os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
# import time
# import numpy as np
# import tensorflow as tf
# import logging
# import datetime
# import mlflow
# import mlflow.keras
# import psutil

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
# from preprocessing import create_sequences

# import config

# # Configure logging with datetime
# logging.basicConfig(
#     filename=os.path.join(config.log_dir, 'training.log'),
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# def train_model(df_log, market, seq_length=12, epochs=100, batch_size=64):
#     # Prepare data for LSTM
#     data = df_log[['o_rice', 'h_rice', 'l_rice', 'c_rice']].values
#     X, y = create_sequences(data, seq_length)

#     # Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     # Build the LSTM model
#     model = Sequential()
#     model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(Dropout(0.3))
#     model.add(LSTM(units=100, return_sequences=False))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=1))

#     # Compile the model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

#     # Start MLflow tracking
#     mlflow.set_tracking_uri(config.mlflow_tracking_uri)
#     with mlflow.start_run(run_name=market):
#         time.sleep(15)
#         # Log parameters
#         mlflow.log_param("market", market)
#         mlflow.log_param("epochs", epochs)
#         mlflow.log_param("batch_size", batch_size)

#         # Track system metrics
#         mlflow.log_metric("cpu_percent", psutil.cpu_percent(interval=None))
#         mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)
        
#         # Train the model
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

#         # Save the model
#         model_name = f"lstm_model_{market}.keras"
#         save_model(model, model_name)
        
#         # Log model
#         mlflow.keras.log_model(model, "model")
        
#         # Predict on the test set
#         y_pred = model.predict(X_test)

#         # Calculate and log metrics
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         mse = mean_squared_error(y_test, y_pred)

#         # Calculate mode accuracy (rounded predictions)
#         y_pred_rounded = np.round(y_pred)
#         accuracy = accuracy_score(np.round(y_test), y_pred_rounded)

#         # Log metrics to MLflow
#         mlflow.log_metric("MAE", mae)
#         mlflow.log_metric("RMSE", rmse)
#         mlflow.log_metric("MSE", mse)
#         mlflow.log_metric("Accuracy", accuracy)

#         # Log training and validation loss
#         train_loss = history.history['loss']
#         val_loss = history.history['val_loss']
#         mlflow.log_metric("train_loss", train_loss[-1])
#         mlflow.log_metric("val_loss", val_loss[-1])
        
#         logging.info(f"Market: {market}, MAE: {mae}, RMSE: {rmse}, MSE: {mse}, Accuracy: {accuracy}")
#     # print(mlflow.MlflowClient().get_run(run.info.run_id).data)
#     return model, history, X_test, y_test

# def save_model(model, model_name):
#     model_dir = config.model_dir
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, model_name)
#     model.save(model_path)
#     logging.info(f"Model saved as {model_path}")

#     # Log model summary
#     current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     model_summary_file = os.path.join(config.log_dir, f'model_summary_{current_datetime}.txt')
#     with open(model_summary_file, 'w') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#     logging.info(f"Model summary saved as {model_summary_file}")

import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import time
import numpy as np
import tensorflow as tf
import logging
import datetime
import mlflow
import mlflow.keras
import psutil
import time
import matplotlib.pyplot as plt  # For plotting

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from preprocessing import create_sequences

import config

# Configure logging with datetime
logging.basicConfig(
    filename=os.path.join(config.log_dir, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model(df_log, market, seq_length=12, epochs=50, batch_size=64):
    # Prepare data for LSTM
    data = df_log[['o_rice', 'h_rice', 'l_rice', 'c_rice']].values
    X, y = create_sequences(data, seq_length)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # Start MLflow tracking
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # Start timer
    start_time = time.time()
    
    with mlflow.start_run(run_name=market):
        time.sleep(15)
        # Log parameters
        mlflow.log_param("market", market)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # Track system metrics
        mlflow.log_metric("cpu_percent", psutil.cpu_percent(interval=None))
        mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

        # Save the model
        model_name = f"lstm_model_{market}.keras"
        save_model(model, model_name)
        
        # Log model
        mlflow.keras.log_model(model, "model")
        
        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate and log metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)

        # Calculate mode accuracy (rounded predictions)
        y_pred_rounded = np.round(y_pred)
        accuracy = accuracy_score(np.round(y_test), y_pred_rounded) * 100  # As percentage

        # Log metrics to MLflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("Accuracy_Percentage", accuracy)

        # Log training and validation loss
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        mlflow.log_metric("train_loss", train_loss[-1])
        mlflow.log_metric("val_loss", val_loss[-1])
        
        # Calculate duration and log it
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("Duration_seconds", duration)
        
        # Plot training and validation loss
        plot_loss(history, market)
        
        logging.info(f"Market: {market}, MAE: {mae}, RMSE: {rmse}, MSE: {mse}, Accuracy: {accuracy}%, Duration: {duration} seconds")

    return model, history, X_test, y_test

def save_model(model, model_name):
    model_dir = config.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    logging.info(f"Model saved as {model_path}")

    # Log model summary
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_summary_file = os.path.join(config.log_dir, f'model_summary_{current_datetime}.txt')
    with open(model_summary_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logging.info(f"Model summary saved as {model_summary_file}")

def plot_loss(history, market):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Training and Validation Loss for {market}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(config.log_dir, f'loss_plot_{market}.png')
    plt.savefig(plot_file)
    plt.close()
    mlflow.log_artifact(plot_file)
    logging.info(f"Loss plot saved as {plot_file}")
