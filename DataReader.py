import cv2, glob, os, random
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

def data_generator(img_dir, batch_size, dictionary):
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
            image = cv2.resize(image, (256, 256))

            # Convert to LAB images from images in BGR format
            # OpenCV converts L values into the range [0, 255] by L <- L * 255/100
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            y.append(lab_image[:, :, 1:])
            x.append(lab_image[:, :, 0])

        # load all the data into numpy arrays
        y = np.array(y)
        x = np.array(x).reshape((batch_size, 256, 256, 1))

        binSpace = np.linspace(0, 256, 21)
        weights = []
        cat = []
        bins = [0,8,14,20,25,106,143,139,179,159,189,199,209,219,248,265,259,283,307,323,334,341,380,384,23]
        closestBins = findClosestBins(bins)
        for image_ab in y:
            y_bin = np.digitize(image_ab, binSpace) - 1
            y_bin = y_bin[:, :, 0] * 20 + y_bin[:, :, 1]
            y_cat = to_categorical(y_bin, num_classes = 400)
            y_bin = np.vectorize(dictionary.get)(y_bin)
            cat.append(y_cat)
            weights.append(y_bin)

        cat = np.array(cat)
        weights = np.array(weights).reshape((batch_size, 256, 256, 1))

        yield (x, cat)
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

def findClosestBins(bins):
    binDict = {}
    for b in range(400):
        if b in bins:
            binDict[b] = b
        else:
            minDistance = 1000000
            bin_alpha = 13*(b//20)+6
            bin_beta = 13*(b%20)+6
            for otherBin in bins:
                otherBin_alpha = 13 * (otherBin // 20) + 6
                otherBin_beta = 13 * (otherBin % 20) + 6
                if (otherBin_alpha - bin_alpha)*(otherBin_alpha - bin_alpha) + (otherBin_beta - bin_beta)*(otherBin_beta - bin_beta) < minDistance:
                    minDistance = (otherBin_alpha - bin_alpha)*(otherBin_alpha - bin_alpha) + (otherBin_beta - bin_beta)*(otherBin_beta - bin_beta)
                    binDict[b] = otherBin
    return binDict
