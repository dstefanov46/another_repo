import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import he_normal, glorot_normal
#from tensorflow import math

def callbacks_func(folder, save_freq="epoch", verbose=1, save_only_best=True, mode="min"):
    """I created this func in order to use the same callbacks accross notebook.
        In case of creating one checkpoint object, the last learned weights stay in object,
        thus it is crucial to allways initialize it.
    
    """
    if save_only_best:
        filename = "temp_weights.h5"
    else: 
        filename = "weights/temp_weights_{epoch:02d}.h5"
    
    checkpoint = ModelCheckpoint(folder + filename, 
                                 monitor="val_loss",
                                 verbose=verbose, 
                                 save_best_only=save_only_best, 
                                 mode=mode,  # če sejvaš uteži z najmanjšim al največjim lossom
                                 
                                 # neki ga zjebe če nastavim spodnji parameter, tudi če ga nastavim na default
                                 # original je bilo to period, ampak sedaj javla da je deprecated
                                
                                # save_freq=save_freq  # na koliko epochov save-a, pri MLPjih ko imaš veliko epochov, ima to smisel
                                )
    return [checkpoint]


def save_model(model, filepath):
    model_json = model.to_json()
    with open(filepath + "model.json", "w") as json_file:
        json_file.write(model_json)
        
        
def load_json_model(filepath):
    json_file = open(filepath + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model