from keras.models import Model
from keras.layers import *


def model_cnn_3DPoints(input_shape, nfilters=3):

    img_input = Input(shape=input_shape)
    x = GaussianNoise(.05)(img_input)

    x = Convolution2D(nfilters, 3, strides=1, padding='valid', activation='relu', name='block1_conv1')(x)
    x = Convolution2D(nfilters, 3, strides=1, padding='valid', activation='relu', name='block1_conv2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(3, activation='relu')(x)

    model = Model(img_input, x, name='Kind')
    return model


