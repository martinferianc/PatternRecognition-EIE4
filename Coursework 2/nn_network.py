import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Lambda
import os
from keras import regularizers

import keras.backend as K

SHAPE = (2048,)

MODEL_FILEPATH='weights/.weights.best.hdf5'

from nn_preprocess import load_data


def abs(tensors):
    """
    Compute the absolute value between the tensors

    Parameters
    ----------
    tensors: tensors
        Two input tensors

    Returns
    -------
    tensor: tesor
        Absolute value of a tensor
    """
    out = K.abs(tensors[0]-tensors[1])
    return out

def net():
    """
    Neural Network

    We have a 2 input neural network whose first layer
    takses the absolute difference between the inputs
    and then has a single fully connected layer and an output node

    Our optimiser is nADAM and the loss function is defined as binary cross-entropy

    Returns
    -------
    model: Keras model
    """
    Input_x = Input(shape=SHAPE, name='input1')
    Input_y = Input(shape=SHAPE, name ='input2')

    model = Lambda(abs, SHAPE)([Input_x, Input_y])

    model = Dense(400, activation='relu',  kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.0001) )(model)
    model = Dropout(0.3)(model)
    model = BatchNormalization()(model)

    out = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=[Input_x, Input_y], outputs=out)
    return model

def train(X_train,Y_train, values_train,X_validation,Y_validation, values_validation):
    """
    Neural Network trainer

    The stopping conditions are automatic, if the validation loss
    does not improve for 5 consecutive epochs terminate the
    training early

    Parameters
    ----------
    X_train: numpy array
        Input features for training

    Y_train: numpy array
        Input features for training

    values_train: numpy array
        Penalty values for training pairs

    X_validation: numpy array
        Input features for validation

    Y_validation: numpy array
        Input features for validation

    values_validation: numpy array
        Penalty values for validation pairs

    Returns
    -------
    model: Keras model
        Trained model
    """
    model = net()
    print(model.summary())
    opt = keras.optimizers.nadam()
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=["binary_crossentropy"])

    batch_size = 64
    early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 3, mode= 'auto')
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_FILEPATH, monitor='val_loss', verbose=3, save_best_only=True, mode='auto')
    callbacks_list = [early_stopping, checkpoint]

    history = model.fit([X_train,Y_train], values_train, batch_size = batch_size, epochs =50,
          callbacks=callbacks_list,  validation_data=([X_validation, Y_validation], values_validation))
    return model

def load_model():
    """
    Neural Network loader

    Loads the trained network

    Returns
    -------
    model: Keras model
        Trained model
    """
    model = net()
    if os.path.exists(MODEL_FILEPATH):
        model.load_weights(MODEL_FILEPATH)
    else:
        raise Exception("No model was trained, wieghts could not be found!")
    return model

if __name__ == '__main__':
    X_train,Y_train, values_train, X_validation, Y_validation, values_validation = load_data(False)
    model = train(X_train,Y_train, values_train, X_validation, Y_validation, values_validation)

    print(model.predict([X_validation[:10,:], Y_validation[:10,:]]))
    print(values_validation[:10])
