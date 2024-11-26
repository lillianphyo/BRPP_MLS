import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from preprocessing import inverse_log_transform
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Configure logging with datetime
logging.basicConfig(filename=os.path.join('logs', 'evaluation.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, history, X_test, y_test, csv_file):
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    logging.info(f'Test Loss for {os.path.basename(csv_file)}: {test_loss}')

    # Predictions
    predictions = model.predict(X_test)

    # Inverse transform the predictions and test values
    y_test_original = inverse_log_transform(pd.DataFrame(y_test, columns=['c_rice']), ['c_rice'])
    predictions_original = inverse_log_transform(pd.DataFrame(predictions, columns=['c_rice']), ['c_rice'])

    # Calculate MAE, RMSE, and MSE
    mae = mean_absolute_error(y_test_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mse = mean_squared_error(y_test_original, predictions_original)

    logging.info(f'MAE: {mae}, RMSE: {rmse}, MSE: {mse}')

    # Plot the training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for {os.path.basename(csv_file)}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config.plot_dir, f"loss_plot_{os.path.basename(csv_file).split('.')[0]}.png"))
    logging.info(f"Loss plot saved as loss_plot_{os.path.basename(csv_file).split('.')[0]}.png")

    # Plot the predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original, label='Actual Prices')
    plt.plot(predictions_original, label='Predicted Prices')
    plt.title(f'Actual vs Predicted Prices for {os.path.basename(csv_file)}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(config.plot_dir, f"predicted_vs_actual_{os.path.basename(csv_file).split('.')[0]}.png"))
    logging.info(f"Plot saved as predicted_vs_actual_{os.path.basename(csv_file).split('.')[0]}.png")
    plt.close()
