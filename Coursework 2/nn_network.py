import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Activation, Dropout, BatchNormalization, Conv2D, Lambda, add, LeakyReLU, MaxPool2D,GlobalAveragePooling2D
import os

H = 64
W = 32
SHAPE = (H,W,1)

MODEL_FILEPATH='weights/.weights.best.hdf5'

cardinality = 16


from nn_preprocess import load_data

def residual_network():

    Input_x = Input(shape=SHAPE, name='input1')
    Input_y = Input(shape=SHAPE, name ='input2')

    x = concatenate([Input_x, Input_y],axis=-1)

    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:

        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        y = add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = LeakyReLU()(y)

        return y

    # conv1
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1)(x)
    x = Activation('relu')(x)
    model = Model(inputs=[Input_x, Input_y], outputs=x)
    return model
"""
def net():
    Input_x = Input(shape=SHAPE, name='input1')
    Input_y = Input(shape=SHAPE, name ='input2')

    model = concatenate([Input_x, Input_y])
    model = Dense(100)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.3)(model)

    model = Dense(50)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.3)(model)

    model = Dense(25)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.3)(model)

    model = Dense(10)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.3)(model)

    out = Dense(1, activation='relu')(model)

    model = Model(inputs=[Input_x, Input_y], outputs=out)
    return model
"""
def train(X_train,Y_train, values_train,X_validation,Y_validation, values_validation):
    model = residual_network()
    opt = keras.optimizers.nadam()
    model.compile(optimizer=opt,loss='mean_squared_error',metrics=["mean_squared_error"])

    batch_size = 64
    early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 3, mode= 'auto')
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_FILEPATH, monitor='val_loss', verbose=3, save_best_only=True, mode='auto')
    callbacks_list = [early_stopping, checkpoint]

    history = model.fit([X_train,Y_train], values_train, batch_size = batch_size, epochs =50,
          callbacks=callbacks_list,  validation_data=([X_validation, Y_validation], values_validation))

def load_model():
    model = residual_network()
    if os.path.exists(MODEL_FILEPATH):
        model.load_weights(MODEL_FILEPATH)
    else:
        raise Exception("No model was trained, wieghts could not be found!")
    return model

def metric(x,y,model):
    return model.predict([x.reshape(1, -1),y.reshape(1, -1)], verbose=0)

if __name__ == '__main__':
    X_train,Y_train, values_train, X_validation, Y_validation, values_validation = load_data()
    X_train = np.reshape(X_train, (X_train.shape[0],H,W,1))
    Y_train = np.reshape(Y_train, (Y_train.shape[0],H,W,1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0],H,W,1))
    Y_validation = np.reshape(Y_validation, (Y_validation.shape[0],H,W,1))
    train(X_train,Y_train, values_train, X_validation, Y_validation, values_validation)
