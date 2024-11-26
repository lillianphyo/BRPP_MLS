
#to tune predicted value (data scaling)
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import json
from preprocessing import log_transform, inverse_log_transform
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from keras import backend as K
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configuration file path
CONFIG_FILE = 'config.json'

# Load configuration
def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

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

# Load and preprocess data
def prepare_data(o_rice, h_rice, l_rice, c_rice):
    # Convert input features to DataFrame
    input_data = pd.DataFrame({
        'o_rice': [o_rice],
        'h_rice': [h_rice],
        'l_rice': [l_rice],
        'c_rice': [c_rice]
    })

    # Apply log transformation
    input_log_transformed = log_transform(input_data, ['o_rice', 'h_rice', 'l_rice', 'c_rice'])
    
    # Reshape data to match model input requirements
    return np.array([input_log_transformed.values])

# Load the model
def load_model_keras(model_path):
    # Load model using Keras 3 method for SavedModel format
    return tf.keras.models.load_model(model_path, custom_objects={'TFSMLayer': TFSMLayer})

# Make prediction
def predict_rice_price(geo_id, o_rice, h_rice, l_rice, c_rice):
    # Load configuration
    config = load_config()

    # Load the model
    model_path = os.path.join(config['model_dir'], f"lstm_model_{geo_id}.savedmodel")
    model = load_model_keras(model_path)

    # Prepare data for prediction
    input_data = prepare_data(o_rice, h_rice, l_rice, c_rice)

    # Predict
    predictions = model.predict(input_data)
    
    # Inverse transform the prediction
    predicted_price = inverse_log_transform(pd.DataFrame(predictions, columns=['c_rice']), ['c_rice'])
    
    return predicted_price['c_rice'].values[0]

def main():
    # Example values
    geo_id = 'yangon'
    o_rice = 100
    h_rice = 105
    l_rice = 95
    c_rice = 100

    predicted_price = predict_rice_price(geo_id, o_rice, h_rice, l_rice, c_rice)
    print(f"Predicted Rice Price: {predicted_price}")

if __name__ == "__main__":
    main()
