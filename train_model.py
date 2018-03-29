import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import *
from utils import *


image_dir = os.path.abspath(os.getcwd() + '/../')
label_dir = os.path.abspath(os.getcwd() + '/../Annotations/')


label_array = pd.read_csv(label_dir + '/skirt_length_labels.csv', sep=',', header=None)
labels = list(label_array[2])


img = Image.open(image_dir +'/Images/skirt_length_labels/c17475177b78a0bdfe7741ad0a0cce4d.jpg')


imgs = [Image.open(image_dir +'/'+ name) for name in label_array[0]]
imgs = [resize_512(img) for img in imgs]
imgs = [img_to_array(img) for img in imgs]


