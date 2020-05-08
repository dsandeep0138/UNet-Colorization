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

	def huber_loss(y_true, y_pred, clip_delta=1.0):
		error = y_true - y_pred
		cond = tf.keras.backend.abs(error) < clip_delta
		squared_loss = 0.5 * tf.keras.backend.square(error)
		linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

		return tf.where(cond, squared_loss, linear_loss)

	gen_model.compile(optimizer=optimizers.Adam(lr=0.001),
			loss='mse')

	disc_model.compile(optimizer=optimizers.Adam(lr=0.001),
			loss='binary_crossentropy',
			metrics=["accuracy"])

	#gan_input = Input(shape=(256, 256, 1))
	gan_input = Input(shape=(32, 32, 1))
	img_color = gen_model(gan_input)
	disc_model.trainable = False

	real_or_fake = disc_model(img_color)
	gan = Model(gan_input, real_or_fake)
	gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001))
	gan.summary()

	print(x_train.shape)
	print(y_train.shape)
	train(x_train, y_train, x_test, y_test, 1000, gan, gen_model, disc_model, properties)

	'''
	log_directory = properties.log_dir+str(time())    
	tensorboard = TensorBoard(log_dir=log_directory)

	filepath = properties.model_dir+ "weights-{epoch:02d}-{acc:.2f}.h5"
	checkpoint = ModelCheckpoint(filepath,
				monitor='acc',
				verbose=1,
				save_best_only=True,
				mode='max')
	y_train_new = formbins(y_train)
	unique, counts = np.unique(y_train_new, return_counts=True)
	dictionary = dict(zip(unique, counts))
	print(dictionary)    
	#for key in dictionary:
		#a = key//20
		#b = key%20
		#alpha = np.full((256, 256), 13*a + 7)
		#beta = np.full((256, 256), 13*b + 7)
		#L = np.full((256, 256), 50)
		#print(13*a + 7, 13*b + 7) 
		#image_lab = np.dstack((L, alpha, beta))
		#print(image_lab.shape)
		#image_lab = image_lab.astype(np.uint8)
		#image = cv2.cvtColor(image_lab, cv2.COLOR_Lab2RGB)
		#outputfileName = properties.results_dir+str(key)+'.jpg'
		#cv2.imwrite(outputfileName, image)
	#fit/fit_generator giving OOM when batch_size is high
	model.fit(x_train, y_train,
		epochs=100,
		batch_size=16,
		callbacks=[tensorboard, checkpoint],
		verbose=1)

	#model.fit_generator(data_generator(data_dir, 10),
			#steps_per_epoch=len(os.listdir(data_dir)) // 10,
			#epochs=1,
			#callbacks=[tensorboard, checkpoint],
			#verbose=1)		

	#Burn! Burn! Burn! How do I know the corresponding 3rd channel for each prediction?
	#y_pred = model.predict_generator(data_generator(test_dir, 10),
	#		steps=len(os.listdir(test_dir)) // 10,
	#		verbose=1)
	'''

	for i in range(0, len(x_test)):
		y_pred = gen_model.predict(x_test[i].reshape(1, 32, 32, 1))
		y_pred = y_pred.reshape(32, 32, 2)
		y_pred = np.dstack((x_test[i], y_pred))
		y_pred = (y_pred + 1) * 127

		#y_pred = gen_model.predict(x_test[i].reshape(1, 256, 256, 1))
		#y_pred = np.dstack((x_test[i], y_pred.reshape(256, 256, 2)))
		#y_pred = (y_pred  + 1) * 127.5

		#y_pred[:, :, 0] = np.clip(y_pred[:, :, 0], 0, 100)
		#y_pred[:, :, 1] = np.clip(y_pred[:, :, 1], -127, 128)
		#y_pred[:, :, 2] = np.clip(y_pred[:, :, 2], -127, 128)

		#print(y_pred)
		y_pred = y_pred.astype(np.uint8)
		image = cv2.cvtColor(y_pred, cv2.COLOR_LAB2RGB)
		#image = color.lab2rgb(y_pred)
		#image = sni.zoom(image, (256, 256, 3))
		outputfileName = properties.results_dir+str(i)+'.jpg'        
		cv2.imwrite(outputfileName, image)

'''
def train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB, gan, gen_model, disc_model):
	generated_images = gen_model.predict(X_train_L)
	X_train = np.concatenate((X_train_AB, generated_images))
	n = len(X_train_L)
	y_train = np.array([[1]] * n + [[0]] * n)
	rand_arr = np.arange(len(X_train))
	np.random.shuffle(rand_arr)
	X_train = X_train[rand_arr]
	y_train = y_train[rand_arr]

	test_generated_images = gen_model.predict(X_test_L)
	X_test = np.concatenate((X_test_AB, test_generated_images))
	n = len(X_test_L)
	y_test = np.array([[1]] * n + [[0]] * n)
	rand_arr = np.arange(len(X_test))
	np.random.shuffle(rand_arr)
	X_test = X_test[rand_arr]
	y_test = y_test[rand_arr]

        disc_model.fit(x=X_train, y=y_train, epochs=1)
	metrics = disc_model.evaluate(x=X_test, y=y_test)
	print('\n accuracy:',metrics[1])
	#if metrics[1] < .90:
	#	train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB, gan, gen_model, disc_model)
'''

def train(X_train_L, X_train_AB, X_test_L, X_test_AB, epochs, gan, gen_model, disc_model, properties):

	g_losses = []
	d_losses = []
	d_acc = []

	X_train = X_train_L
	n = len(X_train)
	y_train_fake = np.zeros([n,1])
	y_train_real = np.ones([n,1])

	for e in range(epochs):
            print("Epoch %d" % e)
            X_train, X_train_AB = next(data_generator(properties.data_dir, 128))
            #X_train = X_train[0:32, :, :, :]
            #X_train_AB = X_train_AB[0:32, :, :, :]
            
            batch_size = len(X_train)
            #y_train_fake = np.zeros([n,1])
            #y_train_real = np.ones([n,1])

	    #generate images
	    #np.random.shuffle(X_train)
            #noise = (np.random.rand(n, 256, 256, 1) * 255).astype(np.uint8)
            #print(noise)
            
	    #generated_images = gen_model.predict(X_train, verbose=1)
            generated_images = gen_model.predict(X_train, verbose=1)
            generated_images = generated_images.astype(np.uint8)
            real_fake = np.concatenate((X_train_AB, generated_images))
            real_labels = [1] * batch_size
            fake_labels = [0] * batch_size
            
	    #np.random.shuffle(X_train_AB)

	    #Train Discriminator
	    #d_loss = disc_model.fit(x=X_train_AB, y=y_train_real, batch_size=16, epochs=1)
	    #if e % 3 == 2:
            #noise = (np.random.rand(n, 256, 256, 2) * 255).astype(np.uint8)
            #d_loss = disc_model.fit(x=noise, y=y_train_fake, batch_size=16, epochs=1)
                    
	    #d_loss = disc_model.fit(x=generated_images, y=y_train_fake, batch_size=16, epochs=1)

            disc_model.trainable = True
            #gen_model.trainable = False
            d_loss = disc_model.train_on_batch(real_fake, real_labels + fake_labels)
            #print("Discriminator loss: %f" % (d_loss))
            print(d_loss)
            #disc_model.train_on_batch(X_train_AB, y_train_real)
            #disc_model.train_on_batch(generated_images, y_train_fake)
            
            disc_model.trainable = False
            #gen_model.trainable = True
            #noise = (np.random.rand(n, 256, 256, 1) * 255).astype(np.uint8)
            
            #X_train, X_train_AB = next(data_generator(properties.data_dir, 32))
            gan_metrics = gan.train_on_batch(X_train, real_labels)
            print("GAN loss: %f" % (gan_metrics))

        #d_losses.append(d_loss.history['loss'][-1])
		#d_acc.append(d_loss.history['acc'][-1])
		#print('d_loss:', d_loss.history['loss'][-1])
		# print("Discriminator Accuracy: ", disc_acc)

		#train GAN on grayscaled images , set output class to colorized
		#g_loss = gan.fit(x=X_train, y=y_train_real, batch_size=16, epochs=1)

		#Record Losses/Acc
		#g_losses.append(g_loss.history['loss'][-1])
		#print('Generator Loss: ', g_loss.history['loss'][-1])
		#disc_acc = d_loss.history['acc'][-1]

		# Retrain Discriminator if accuracy drops below .8
		#if disc_acc < .8 and e < (epochs / 2):
		#	train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB, gan, gen_model, disc_model)
		#if e % 5 == 4:
		#	print(e + 1,"batches done")

def formbins(y_train):
    bins = np.linspace(0,260,21)
    y_train_new = []
    for image_ab in y_train:
        #Create bins - each bin size is kept as 13 so there are roughly 20 bins from 0 to 255
        #bins = ([  0.,  13.,  26.,  39.,  52.,  65.,  78.,  91., 104., 117., 130.,
        #143., 156., 169., 182., 195., 208., 221., 234., 247., 260.])
        y_train_bin = np.digitize(image_ab, bins)-1 #returns a value in 0 to 19
        #Bin value is a*20 + b
        #To extract a & b from value b = bin%20; a = bin/20
        y_train_bin = y_train_bin[:,:,0]*20+y_train_bin[:,:,1] 
        y_train_new.append(y_train_bin)
    return np.transpose(np.array(y_train_new),(1,2,0))
              
if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()

