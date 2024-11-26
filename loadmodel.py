from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from pathlib import Path

from keras import backend as K # Estrutura de manipulação de tensores simbólicos

""" IoU """
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0) # Média de um tensor, ao longo do eixo especificado
    return iou

""" Dice Coefficient """
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

""" Dice Coefficient Loss """
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

scope = {'iou': iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}
path_model = 'data/model/lstm_model_yangon.keras'

with CustomObjectScope(scope):
    modelo = load_model(Path(path_model))