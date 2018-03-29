from keras.preprocessing.image import ImageDataGenerator

prep = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2)

