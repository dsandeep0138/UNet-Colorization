import keras
from DataReader import load_data


def main():
    data_dir = './data/beach'
    test_dir = './data/test'

    x_train, y_train, x_test, y_test = load_data(data_dir, test_dir)


if __name__ == "__main__":
    main()
