from DataReader import load_data
from Network import UNetRegressor


def main():
	data_dir = './data/beach'
	test_dir = './data/test'

	#x_train, y_train, x_test, y_test = load_data(data_dir, test_dir)

	model = UNetRegressor(64, 3)
	model.build_model()


if __name__ == "__main__":
	main()
