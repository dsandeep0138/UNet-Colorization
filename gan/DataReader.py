import cv2
import glob
import numpy as np
import os
import random
from skimage import color

def data_generator(img_dir, batch_size):
    # Get all the image file names in data_dir
    filenames = glob.glob(img_dir + "/*")

    # Set seed for reproducibility
    random.seed(97)

    counter = 0
    while True:
        x, y = ([] for _ in range(2))

        # Randomly shuffle the data
        random.shuffle(filenames)

        if (counter + 1) * batch_size >= len(filenames):
            counter = 0

        for i in range(batch_size):
            image = cv2.imread(filenames[counter * batch_size + i])

            # Resize all images to (256, 256, 3)
            #image = cv2.resize(image, (256, 256))
            image = cv2.resize(image, (32, 32))

            # Convert to LAB images from images in BGR format
            # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            '''
            lab_image = color.rgb2lab(image)
            lab_image[:, :, 0] = 2 * lab_image[:, :, 0] / 100 - 1
            lab_image[:, :, 1:] = lab_image[:, :, 1:] / 127
            '''

            y.append(lab_image[:, :, 1:])
            x.append(lab_image[:, :, 0])

        # load all the data into numpy arrays
        #y = np.array(y)
        #x = np.array(x).reshape((batch_size, 32, 32, 1))

        y = (np.asarray(y, dtype=np.float32) - 127.5) / 127.5
        x = (np.asarray(x, dtype=np.float32) - 127.5) / 127.5
        x = x.reshape((batch_size, 32, 32, 1))

        yield (x, y)
        counter += 1


def cifar_data_generator(X, y, batch_size):
	num_batches = X.shape[0] / batch_size
	batch_index = 0

	while True:
		if batch_index + batch_size >= X.shape[0]:
			batch_index = 0

		for j in range(num_batches):
			batch_index += batch_size
			X_batch = X[batch_index:batch_index + batch_size]
			y_batch = y[batch_index:batch_index + batch_size]
		
			yield i, j


def load_cifar_data(data_dir):
	y_train = []

	for i in range(1, 6):
		train_data_dir = data_dir + "/data_batch_{}".format(i)
		with open(train_data_dir, 'rb') as f:
			train_data_dir_dict = pickle.load(f, encoding='bytes')
		if i == 1:
			x_train = train_data_dir_dict[b'data']
		else:
			x_train = np.vstack((x_train, train_data_dir_dict[b'data']))
		y_train += train_data_dir_dict[b'labels']

	y_train = np.array(y_train)

	test_data_dir = data_dir + "/test_batch"
	with open(test_data_dir, 'rb') as f:
		test_data_dir_dict = pickle.load(f, encoding='bytes')
	x_test = test_data_dir_dict[b'data']
	y_test = test_data_dir_dict[b'labels']
	y_test = np.array(y_test)

	x_train = x_train.reshape((len(x_train), 32, 32, 3))
	x_test = x_test.reshape((len(x_test), 32, 32, 3))

	return x_train, y_train, x_test, y_test


def load_data(data_dir, test_dir):
    x_train, y_train, x_test, y_test = ([] for _ in range(4))

    # Get all the image file names in data_dir
    filenames = glob.glob(data_dir + "/*")

    # Random shuffle the data and set seed for reproducibility
    random.seed(97)
    random.shuffle(filenames)

    for i, filename in enumerate(filenames):
        image = cv2.imread(filename)

        # Resize all images to (256, 256, 3)
        image = cv2.resize(image, (32, 32))
        #image = cv2.resize(image, (256, 256))

        # Convert to LAB images from images in BGR format
        # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        #lab_image = color.rgb2lab(image)
        #lab_image[:, :, 0] = 2 * lab_image[:, :, 0] / 100 - 1
        #lab_image[:, :, 1:] = lab_image[:, :, 1:] / 127

        y_train.append(lab_image[:, :, 1:])
        x_train.append(lab_image[:, :, 0])

    # load all the data into numpy arrays
    #y_train = np.array(y_train)
    #x_train = np.array(x_train).reshape((len(x_train), 32, 32, 1))
    y_train = (np.asarray(y_train, dtype=np.float32) - 127.5) / 127.5
    x_train = (np.asarray(x_train, dtype=np.float32) - 127.5) / 127.5
    #y_train = np.asarray(y_train, dtype=np.float32) / 255.
    #x_train = np.asarray(x_train, dtype=np.float32) / 255.
    #x_train = x_train.reshape((len(x_train), 32, 32, 1))

    # Similar processing for test data
    test_files = glob.glob(test_dir + "/*")

    for test_file in test_files:
        image = cv2.imread(test_file)
        #image = cv2.resize(image, (256, 256))
        image = cv2.resize(image, (32, 32))
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        '''
        lab_image = color.rgb2lab(image)
        lab_image[:, :, 0] = 2 * lab_image[:, :, 0] / 100 - 1
        lab_image[:, :, 1:] = lab_image[:, :, 1:] / 127

        lab_image = color.rgb2lab(image)
        lab_image[:, :, 0] = lab_image[:, :, 0] * 255 / 100
        lab_image[:, :, 1] += 128
        lab_image[:, :, 2] += 128 
        lab_image /= 255
        '''

        y_test.append(lab_image[:, :, 1:])
        x_test.append(lab_image[:, :, 0])

    #y_test = np.array(y_test)
    #x_test = np.array(x_test).reshape((len(x_test), 32, 32, 1))
    y_test = (np.asarray(y_test, dtype=np.float32) - 127.5) / 127.5
    x_test = (np.asarray(x_test, dtype=np.float32) - 127.5) / 127.5
    x_test = x_test.reshape((len(x_test), 32, 32, 1))

    return x_train, y_train, x_test, y_test
