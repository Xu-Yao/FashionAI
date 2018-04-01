import tensorflow as tf

def parse_jpeg(filename, label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)
	image_resized = 1./255*tf.image.resize_images(image_decoded, [128, 128])
	return image_resized, label


