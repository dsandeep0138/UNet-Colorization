import cv2
import glob
import numpy as np
import os
import random
import shutil


def data_generator(img_dir, batch_size):
    # Get all the image file names in data_dir
    filenames = glob.glob(data_dir + "/*")

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
            image = cv2.imread(filenames[i])

            # Resize all images to (256, 256, 3)
            image = cv2.resize(image, (256, 256))
            y.append(image)

            # Convert to BGR from RGB
            image = image[:, :, ::-1]

            # Convert to LAB images from images in BGR format
            # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            L, _, _ = cv2.split(lab_image)
            x.append(L)

        # load all the data into numpy arrays
        y = np.array(y)
        x = np.array(x)

        yield (x, y)
        counter += batch_size


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
