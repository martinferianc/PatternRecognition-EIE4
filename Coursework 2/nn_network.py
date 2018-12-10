import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Activation, Dropout, BatchNormalization, Conv2D, Lambda, add, LeakyReLU, MaxPool2D,GlobalAveragePooling2D
import os

H = 64
W = 32
SHAPE = (2048,)

MODEL_FILEPATH='weights/.weights.best.hdf5'


from nn_preprocess import load_data

def net():
    Input_x = Input(shape=SHAPE, name='input1')
    Input_y = Input(shape=SHAPE, name ='input2')

    model = concatenate([Input_x, Input_y])
    model = Dense(100)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.3)(model)

    out = Dense(1, activation='relu')(model)

    model = Model(inputs=[Input_x, Input_y], outputs=out)
    return model

def train(X_train,Y_train, values_train,X_validation,Y_validation, values_validation):
    model = net()
    opt = keras.optimizers.nadam()
    model.compile(optimizer=opt,loss='mean_squared_error',metrics=["mean_squared_error"])

    batch_size = 64
    early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 3, mode= 'auto')
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_FILEPATH, monitor='val_loss', verbose=3, save_best_only=True, mode='auto')
    callbacks_list = [early_stopping, checkpoint]

    history = model.fit([X_train,Y_train], values_train, batch_size = batch_size, epochs =50,
          callbacks=callbacks_list,  validation_data=([X_validation, Y_validation], values_validation))

def load_model():
    model = net()
    if os.path.exists(MODEL_FILEPATH):
        model.load_weights(MODEL_FILEPATH)
    else:
        raise Exception("No model was trained, wieghts could not be found!")
    return model

def metric(x,y,model):
    return model.predict([x.reshape(1, -1),y.reshape(1, -1)], verbose=0)

if __name__ == '__main__':
    X_train,Y_train, values_train, X_validation, Y_validation, values_validation = load_data(False)
    train(X_train,Y_train, values_train, X_validation, Y_validation, values_validation)
