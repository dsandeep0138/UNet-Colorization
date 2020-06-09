import cv2, gc, os, sys
import numpy as np
import tensorflow as tf
from DataReader import load_data, data_generator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from Network import GanGenerator, GanDiscriminator
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

    gen_model = GanGenerator(64, 3).build_model()
    disc_model = GanDiscriminator(64, 3).build_model()

    disc_model.compile(optimizer=optimizers.Adam(lr=0.003, beta_1=0.5),
		       loss='binary_crossentropy',
		       metrics=["accuracy"])

    gan_input = Input(shape=(256, 256, 1))

    disc_model.trainable = False
    img_color = gen_model(gan_input)
    real_or_fake = disc_model(img_color)
    gan = Model(gan_input, real_or_fake)
    gan.compile(optimizer=optimizers.Adam(lr=0.0003, beta_1=0.5),
		loss='binary_crossentropy',
		metrics=["accuracy"])

    gan.summary()

    gc.collect()

    train(x_train, y_train, 50, gan, gen_model, disc_model, properties)

    for i in range(0, len(x_test)):
        y_pred = gen_model.predict(x_test[i].reshape(1, 256, 256, 1))
        y_pred = np.dstack((x_test[i], y_pred.reshape(256, 256, 2)))
        y_pred = (y_pred + 1.) * 127.5

        y_pred = y_pred.astype(np.uint8)
        image = cv2.cvtColor(y_pred, cv2.COLOR_LAB2RGB)
        outputfileName = properties.results_dir + str(i) + '.png'
        cv2.imwrite(outputfileName, image)


def train(X_train_L, X_train_AB, epochs, gan, gen_model, disc_model, properties):
    n = X_train_L.shape[0] // 8

    X_train = np.expand_dims(X_train_L, axis=-1)
    y_train_fake = np.zeros([n, 1])
    y_train_real = np.ones([n, 1])
    y_real_fake = np.zeros([2 * n, 1])
    y_real_fake[:n] = np.random.uniform(low=0.7, high=1, size=(n, 1))

    #np.random.seed(97)

    for epoch in range(1, epochs + 1):
        print("Epoch %d" % epoch)

        np.random.shuffle(X_train)
        np.random.shuffle(X_train_AB)

        if epoch % 3 == 0:
            noise = np.random.uniform(-1, 1, (n, 256, 256, 1))
            generated_images = gen_model.predict(noise, verbose=1)
        else:
            generated_images = gen_model.predict(X_train[:n], verbose=1)

        disc_model.trainable = True
        real_fake = np.concatenate((X_train_AB[:n], generated_images))
        d_loss = disc_model.fit(x=real_fake, y=y_real_fake, batch_size=16, epochs=1)
        gc.collect()

        disc_model.trainable = False
        gan_metrics = gan.fit(x=X_train[:2 * n], y=np.ones([2 * n, 1]), batch_size=16, epochs=1)
        gc.collect()


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
