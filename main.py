from DataReader import load_data, data_generator
from Network import UNetRegressor
from keras import optimizers
import os

def main():
	data_dir = './data/beach'
	test_dir = './data/test'

	#x_train, y_train, x_test, y_test = load_data(data_dir, test_dir)

	model = UNetRegressor(64, 3).build_model()

	model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse')

	model.fit_generator(data_generator(data_dir, 128),
			steps_per_epoch=len(os.listdir(data_dir)) // 128,
			epochs=10,
			verbose=1)		


if __name__ == "__main__":
	main()
