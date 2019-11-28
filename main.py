import cv2
import os
import numpy as np
import tensorflow as tf
import sys
from DataReader import load_data, data_generator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from Network import UNetRegressor
from time import time
from properties import Properties
import keras.backend as K           

def main():
	properties = None    
	if (len(sys.argv) > 1):
		properties = Properties(sys.argv[1])
	else:
		properties = Properties("Local")
    
	x_train, y_train, x_test, y_test = load_data(properties.data_dir, properties.test_dir)

	model = UNetRegressor(64, 3).build_model()

	#def quantile_metric(quantile, y_true, y_pred):
	#	e = y_true - y_pred
	#	metric = K.mean(K.maximum(quantile * e, (quantile - 1) * e), axis=-1)
	#	return metric

	#def loss(y_true, y_pred):
	#	eA = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
	#	eB = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
	#	return K.mean(K.square(eA), axis=-1) + K.mean(K.square(eB), axis=-1)

	def huber_loss(y_true, y_pred, clip_delta=1.0):
		error = y_true - y_pred
		cond = tf.keras.backend.abs(error) < clip_delta
		squared_loss = 0.5 * tf.keras.backend.square(error)
		linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

		return tf.where(cond, squared_loss, linear_loss)

	model.compile(optimizer=optimizers.Adam(lr=0.001),
			loss=lambda y, f: huber_loss(y, f, clip_delta=0.5),
			metrics=["accuracy"])

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

	for i in range(0, len(x_test)):
		y_pred = model.predict(x_test[i].reshape(1, 256, 256, 1))
		y_pred = np.dstack((x_test[i], y_pred.reshape(256, 256, 2)))
		y_pred = y_pred.astype(np.uint8)
		image = cv2.cvtColor(y_pred, cv2.COLOR_LAB2RGB)
		outputfileName = properties.results_dir+str(i)+'.jpg'        
		cv2.imwrite(outputfileName, image)
	

    
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

