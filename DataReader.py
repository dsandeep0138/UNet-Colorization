import cv2
import glob
import numpy as np
import os
from pathlib import Path
import random
import shutil


def train_generator(img_dir, batch_size):
    # Get all the image file names in data_dir
    filenames = glob.glob(data_dir + "/*")

    # Set seed for reproducibility
    random.seed(97)

    while True:
        x_data = []
        y_data = []

        # Randomly shuffle the data
        random.shuffle(filenames)

        for i in range(batch_size):
            print(filename)
            image = cv2.imread(filenames[i])

            # Resize all images to (256, 256, 3)
            image = cv2.resize(image, (256, 256))
            y_data.append(image)

            # Convert to BGR from RGB
            image = image[:, :, ::-1]

            # Convert to LAB images from images in BGR format
            # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            L, _, _ = cv2.split(lab_image)
            x_data.append(L)

        # load all the data into numpy arrays
        y_data = np.array(y_data)
        x_data = np.array(x_data)

        yield (x_data, y_data)


def load_data(data_dir, test_dir):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Get all the image file names in data_dir
    filenames = glob.glob(data_dir + "/*")

    # Random shuffle the data and set seed for reproducibility
    random.seed(97)
    random.shuffle(filenames)

    for i, filename in enumerate(filenames):
        image = cv2.imread(filename)

        # Resize all images to (256, 256, 3)
        image = cv2.resize(image, (256, 256))
        y_train.append(image)

        # Convert to BGR from RGB
        image = image[:, :, ::-1]

        # Convert to LAB images from images in BGR format
        # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab_image)
        x_train.append(L)

    # load all the data into numpy arrays
    y_train = np.array(y_train)
    x_train = np.array(x_train)

    # Similar processing for test data
    test_files = glob.glob(data_dir + "/*")

    for test_file in test_files:
        image = cv2.imread(test_file)
        image = cv2.resize(image, (256, 256))
        y_test.append(image)
        image = image[:, :, ::-1]
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab_image)
        x_test.append(L)

    y_test = np.array(y_test)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test
