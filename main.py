import cv2
import os
import numpy as np
import tensorflow as tf
from DataReader import load_data, data_generator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from Network import UNetRegressor
from time import time

def main():
	data_dir = './data/beach'
	test_dir = './data/test'

	x_train, y_train, x_test, y_test = load_data(data_dir, test_dir)

	model = UNetRegressor(64, 3).build_model()

	model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse', metrics=["accuracy"])

	tensorboard = TensorBoard(log_dir="data/logs/{}".format(time()))

	filepath = "data/models/weights-{epoch:02d}-{acc:.2f}.h5"
	checkpoint = ModelCheckpoint(filepath,
				monitor='acc',
				verbose=1,
				save_best_only=True,
				mode='max')

	#fit/fit_generator giving OOM when batch_size is high
	#model.fit(x_train, y_train,
	#	epochs=1,
	#	batch_size=128,
	#	validation_split=0.2,
	#	callbacks=[tensorboard, checkpoint],
	#	verbose=1)

	model.fit_generator(data_generator(data_dir, 10),
			steps_per_epoch=len(os.listdir(data_dir)) // 10,
			epochs=10,
			callbacks=[tensorboard, checkpoint],
			verbose=1)		

	#Burn! Burn! Burn! How do I know the corresponding 3rd channel for each prediction?
	#y_pred = model.predict_generator(data_generator(test_dir, 10),
	#		steps=len(os.listdir(test_dir)) // 10,
	#		verbose=1)

	for i in range(0, len(x_test)):
		y_pred = model.predict(x_test[i].reshape(1, 256, 256, 1))
		y_pred = np.dstack((x_test[i], y_pred.reshape(256, 256, 2)))
		y_pred = y_pred.astype(np.uint8)
		image = cv2.cvtColor(y_pred, cv2.COLOR_LAB2RGB)
		cv2.imwrite('results/{}.jpg'.format(i), image)


if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
