from DataReader import load_data, data_generator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from Network import UNetClassifier
from time import time
from properties import Properties
import cv2, os, sys
import keras.backend as K
import numpy as np
import tensorflow as tf


def main():
    properties = None
    if (len(sys.argv) > 1):
        properties = Properties(sys.argv[1])
    else:
        properties = Properties("Local")

    x_train, y_train, x_test, y_test = load_data(properties.data_dir, properties.test_dir)

    model = UNetClassifier(num_layers = 3,
			   num_filters = 64,
			   num_classes = 400).build_model()

    def huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = tf.keras.backend.abs(error) < clip_delta
        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

        return tf.where(cond, squared_loss, linear_loss)

    def crossentropy_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)

    y_train_new = formbins(y_train)

    unique, counts = np.unique(y_train_new, return_counts=True)
    weights = []
    weights = 1 - counts/np.sum(counts)
    weights /= np.sum(weights)

    dictionary = dict(zip(unique, weights))
    for i in range(10):
        next(data_generator(properties.data_dir, 10, dictionary))
    
    weightsVector = []
    for i in range(100):
        if i in dictionary:
            weightsVector.append(dictionary[i])
        else:
            weightsVector.append(0)

    loss = weighted_categorical_crossentropy(weightsVector)
    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    log_directory = properties.log_dir + str(time())
    tensorboard = TensorBoard(log_dir=log_directory)

    filepath = properties.model_dir + "weights-{epoch:02d}-{acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    model.fit_generator(data_generator(properties.data_dir, 2, dictionary),
                       steps_per_epoch=len(os.listdir(properties.data_dir)) // 2,
                       epochs=25,
                       callbacks=[tensorboard, checkpoint],
                       verbose=1)

    bins = [0, 8, 14, 20, 25, 106, 143, 139, 179, 159, 189, 199, 209, 219, 248,
            265, 259, 283, 307, 323, 334, 341, 380, 384, 23]
    for i in range(0, len(x_test)):
        y_pred = model.predict(x_test[i].reshape(1, 256, 256, 1))
        if (i == 0):
            for k in range(5):
                for l in range(5):
                    fileName = properties.results_dir + str(k)+str(l) + '.csv'
                    toprint = y_pred[0,k,l,:].reshape(400,)
                    np.savetxt(fileName, toprint, delimiter=',')
        y_pred_bin = y_pred.reshape(256,256,400)
        y_pred_bin = np.argmax(y_pred_bin, axis=2)
        y_pred_alpha = 13*(y_pred_bin//20)+6
        y_pred_beta = 13*(y_pred_bin % 20)+6
        y_pred = np.dstack((x_test[i], y_pred_alpha.reshape(256, 256, 1), y_pred_beta.reshape(256, 256, 1)))
        y_pred = y_pred.astype(np.uint8)
        image = cv2.cvtColor(y_pred, cv2.COLOR_LAB2RGB)
        outputfileName = properties.results_dir + str(i) + '.jpg'
        cv2.imwrite(outputfileName, image)

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def formbins(y_train):
    bins = np.linspace(0, 260, 11)
    y_train_new = []
    for image_ab in y_train:
        # Create bins - each bin size is kept as 13 so there are roughly 20 bins from 0 to 255
        # bins = ([  0.,  13.,  26.,  39.,  52.,  65.,  78.,  91., 104., 117., 130.,
        # 143., 156., 169., 182., 195., 208., 221., 234., 247., 260.])
        y_train_bin = np.digitize(image_ab, bins) - 1  # returns a value in 0 to 19
        # Bin value is a*20 + b
        # To extract a & b from value b = bin%20; a = bin/20
        y_train_bin = y_train_bin[:, :, 0] * 10 + y_train_bin[:, :, 1]
        y_train_new.append(y_train_bin)
    return np.transpose(np.array(y_train_new), (1, 2, 0))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
