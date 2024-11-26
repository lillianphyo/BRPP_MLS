# # utils/predict.py
# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.utils import CustomObjectScope
# from tensorflow.keras.models import load_model
# from keras import backend as K
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config
# from .preprocessing import log_transform, inverse_log_transform
# from pathlib import Path

# def load_trained_model(geo_id):
#     # Load the model
#     """ IoU """
#     def iou(y_true, y_pred, smooth=1):
#         intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#         union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#         iou = K.mean((intersection + smooth) / (union + smooth), axis=0) # Média de um tensor, ao longo do eixo especificado
#         return iou

#     """ Dice Coefficient """
#     def dice_coef(y_true, y_pred, smooth=1):
#         intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#         union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#         return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

#     """ Dice Coefficient Loss """
#     def dice_coef_loss(y_true, y_pred):
#         return 1 - dice_coef(y_true, y_pred)

#     scope = {'iou': iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}
#     with CustomObjectScope(scope):
#         model_path = os.path.join(config.model_dir, f"lstm_model_{geo_id}.keras")
#         model = load_model(Path(model_path))
#         return model

# def predict_rice_price(geo_id, c_rice):
#     # Load model
#     model = load_trained_model(geo_id)

#     # Prepare input data (without scaling)
#     input_data = {
#         'o_rice': c_rice,  # Assuming current price for open, high, low prices
#         'h_rice': c_rice,
#         'l_rice': c_rice,
#         'c_rice': c_rice
#     }
    
#     input_df = pd.DataFrame(input_data, index=[0])

#     # Log-transform the input data
#     input_log_transformed = log_transform(input_df, input_df.columns)

#     # Predict
#     predictions = model.predict(np.array([input_log_transformed.values]))
    
#     # Inverse log transform the prediction
#     predicted_price = inverse_log_transform(pd.DataFrame(predictions, columns=['c_rice']), ['c_rice'])

#     return predicted_price['c_rice'].values[0]

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from .preprocessing import log_transform, inverse_log_transform
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from keras import backend as K
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pathlib import Path

def predict_rice_price(geo_id, o_rice, h_rice, l_rice, c_rice):
    # Load model
    model = load_trained_model(geo_id)

    # Prepare input data
    input_data = {
        'o_rice': [o_rice],
        'h_rice': [h_rice],
        'l_rice': [l_rice]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Log-transform the input data
    input_log_transformed = log_transform(input_df, input_df.columns)

    # Convert the input to a 3D numpy array (batch_size, timesteps, features)
    input_array = np.array([input_log_transformed.values])

    # Predict
    predictions = model.predict(input_array)
    
    # Inverse log transform the prediction
    predicted_price = inverse_log_transform(pd.DataFrame(predictions, columns=['c_rice']), ['c_rice'])

    return float(round(predicted_price['c_rice'].values[0]*100))
    # return predicted_price

def load_trained_model(geo_id):
    # """ IoU """
    # def iou(y_true, y_pred, smooth=1):
    #     intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    #     union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    #     iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    #     return iou

    # """ Dice Coefficient """
    # def dice_coef(y_true, y_pred, smooth=1):
    #     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    #     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    #     return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

    # """ Dice Coefficient Loss """
    # def dice_coef_loss(y_true, y_pred):
    #     return 1 - dice_coef(y_true, y_pred)

    # scope = {'iou': iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}

    # with CustomObjectScope(scope):
    #     # model_path = os.path.join(config.model_dir, f"lstm_model_{geo_id}.keras")
    #     model_path='../data/model/lstm_model_yangon.keras'
    #     model = load_model(Path(model_path))
    #     return model
    # model_path = os.path.join(config.model_dir, f"lstm_model_{geo_id}.keras")
    # print(model_path)
    # model_path='../data/model/lstm_model_mandalay.keras'
    model_path = os.path.join(config.model_dir, f"lstm_model_{geo_id}.keras")
    model = load_model(model_path, compile = True)
    return model

# def main():
#     #506.26	518.56	480.6	505.03

#     geo_id = 'yangon'  # Example geo_id
#     o_rice = 506.26      # Example value for open rice price
#     h_rice = 518.56      # Example value for high rice price
#     l_rice = 480.6      # Example value for low rice price
#     c_rice = 505.03      # Example value for current rice price

#     predicted_price = predict_rice_price(geo_id, o_rice, h_rice, l_rice, c_rice)
#     print(f"Predicted Rice Price: {predicted_price}")

# if __name__ == "__main__":
#     main()
