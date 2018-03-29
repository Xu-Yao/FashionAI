from keras.models import Model
from keras.layers import *


def model_cnn_3DPoints(input_shape,nfilters,classes,dropout_par):

    img_input = Input(shape=input_shape)

    x = Convolution2D(nfilters, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(x)
    x = Convolution2D(nfilters, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(3, activation='relu')(x)

    model = Model(img_input, x, name='Kind')
    return model

