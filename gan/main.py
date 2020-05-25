import cv2
import os
import numpy as np
import tensorflow as tf
import sys
from DataReader import load_data, data_generator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from Network import UNetRegressor, UNetDiscriminator
from time import time
from properties import Properties
import keras.backend as K           
from keras.layers import Input
from keras.models import Model
from skimage import color
import scipy.ndimage.interpolation as sni

def main():
    properties = None    
    if (len(sys.argv) > 1):
        properties = Properties(sys.argv[1])
    else:
        properties = Properties("Local")
    
    x_train, y_train, x_test, y_test = load_data(properties.data_dir, properties.test_dir)

    gen_model = UNetRegressor(64, 3).build_model()
    disc_model = UNetDiscriminator(64, 3).build_model()
    frozen_disc_model = UNetDiscriminator(64, 3).build_model()
    frozen_disc_model.trainable = False

    gen_model.compile(optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5),
                        loss='binary_crossentropy',
		        metrics=["accuracy"])

    disc_model.compile(optimizer=optimizers.Adam(lr=0.00002, beta_1=0.5),
		        loss='binary_crossentropy',
		        metrics=["accuracy"])

    frozen_disc_model.compile(optimizer=optimizers.Adam(lr=0.00002, beta_1=0.5),
		        loss='binary_crossentropy',
		        metrics=["accuracy"])

    gan_input = Input(shape=(256, 256, 1))
    img_color = gen_model(gan_input)
    real_or_fake = frozen_disc_model(img_color)
    gan = Model(gan_input, real_or_fake)
    gan.compile(optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5),
		loss='binary_crossentropy',
		metrics=["accuracy"])
    gan.summary()

    train(x_train, y_train, x_test, y_test, 1000, gan, gen_model, disc_model, properties)

    for i in range(0, len(x_test)):
        y_pred = gen_model.predict(x_test[i].reshape(1, 256, 256, 1))
        y_pred = np.dstack((x_test[i], y_pred.reshape(256, 256, 2)))
        y_pred = (y_pred + 1) * 127.5

        y_pred = y_pred.astype(np.uint8)
        image = cv2.cvtColor(y_pred, cv2.COLOR_LAB2RGB)
        outputfileName = properties.results_dir + str(i) + '.png'
        cv2.imwrite(outputfileName, image)


def train(X_train_L, X_train_AB, X_test_L, X_test_AB, epochs, gan, gen_model, disc_model, properties):
    n = 160

    y_train_real = np.ones([n, 1])
    y_real_fake = np.zeros([2 * n, 1])
    y_real_fake[:n] = 0.9

    for epoch in range(1, epochs + 1):
        print("Epoch %d" % epoch)

        noise = np.random.normal(0, 1, (n, 256, 256, 1))
        generated_images = gen_model.predict(noise, verbose=1)

        np.random.shuffle(X_train_AB)
        real_fake = np.concatenate((X_train_AB[:n], generated_images))

        d_loss = disc_model.fit(x=real_fake, y=y_real_fake, batch_size=32, epochs=1)
        
        noise = np.random.normal(0, 1, (n, 256, 256, 1))
        gan_metrics = gan.fit(x=noise, y=y_train_real, batch_size=32, epochs=1)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()
