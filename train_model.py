from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *

sess = tf.InteractiveSession()

from utils import *
from models import *


nb_epochs = 2
batch_size = 32

image_dir = os.path.abspath(os.getcwd() + '/..')+'/'
label_dir = os.path.abspath(os.getcwd() + '/../Annotations')+'/'
label_array = pd.read_csv(label_dir + 'skirt_length_labels.csv', sep=',', header=None)


img_names = tf.constant((image_dir + label_array[0]))
img_labels = tf.constant(label_array[2])


dataset = tf.data.Dataset.from_tensor_slices((img_names, img_labels))
dataset = dataset.map(parse_jpeg)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(nb_epochs)
iterator = dataset.make_one_shot_iterator()

x, y = iterator.get_next()

net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
print(y.get_shape())

"""
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(nb_epochs):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
"""







