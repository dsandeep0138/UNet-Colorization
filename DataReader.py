import cv2
import glob
import numpy as np
import os
import random
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

def data_generator(img_dir, batch_size, dictionary):
    # Get all the image file names in data_dir
    filenames = glob.glob(img_dir + "/*")

    # Set seed for reproducibility
    random.seed(97)

    #print(weights.shape)

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
            image = cv2.resize(image, (256, 256))

            # Convert to LAB images from images in BGR format
            # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            y.append(lab_image[:, :, 1:])
            x.append(lab_image[:, :, 0])

        # load all the data into numpy arrays
        y = np.array(y)
        x = np.array(x).reshape((batch_size, 256, 256, 1))

        print(y.shape)
        print(y[0][0][0])

        bins = np.linspace(0, 260, 21)
        weights = []
        cat = []
        for image_ab in y:
            y_bin = np.digitize(image_ab, bins) - 1
            y_bin = y_bin[:, :, 0] * 20 + y_bin[:, :, 1]
           
            #print(y_bin.shape)
            #print(y_bin)

            y_cat = to_categorical(y_bin, num_classes=400)
            #print(y_cat.shape)
            #print(y_cat)
 
            y_bin = np.vectorize(dictionary.get)(y_bin)
            #print(y_bin.shape)
            #print(y_bin)
            #exit(0)
            print(y_cat.shape)
            cat.append(y_cat)
            weights.append(y_bin)

        cat = np.array(cat)
        weights = np.array(weights).reshape((batch_size, 256, 256, 1))
        print(cat.shape)
        print(weights.shape)
        weights = np.concatenate((cat, weights), axis=3)
        print(weights.shape)

        yield (x, y)
        counter += 1


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
        image = cv2.resize(image, (256, 256))

        # Convert to LAB images from images in BGR format
        # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        y_train.append(lab_image[:, :, 1:])
        x_train.append(lab_image[:, :, 0])

    # load all the data into numpy arrays
    y_train = np.array(y_train)
    x_train = np.array(x_train).reshape((len(x_train), 256, 256, 1))

    # Similar processing for test data
    test_files = glob.glob(test_dir + "/*")

    for test_file in test_files:
        image = cv2.imread(test_file)
        image = cv2.resize(image, (256, 256))
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        y_test.append(lab_image[:, :, 1:])
        x_test.append(lab_image[:, :, 0])

    y_test = np.array(y_test)
    x_test = np.array(x_test).reshape((len(x_test), 256, 256, 1))

    return x_train, y_train, x_test, y_test
