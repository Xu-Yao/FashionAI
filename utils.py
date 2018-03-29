import tensorflow as tf

def resize_512(image):
	return tf.image.resize_images(image, [512, 512])



