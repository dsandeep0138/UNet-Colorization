class Properties:
    def __init__(self, environment):
        if (environment == "colab"):
            self.data_dir = "/content/cifar/train"
            self.test_dir = "/content/cifar/test"
            self.results_dir = "/content/drive/My Drive/DL-project/data/gan/cifar/results/"
            self.model_dir = "/content/drive/My Drive/UNet-Colorization/models/"
            self.log_dir = "/content/drive/My Drive/UNet-Colorization/data/logs/"
        else :
            #self.data_dir = "./../data/beach"
            #self.test_dir = "./../data/test"
            self.data_dir = "./../data/cifar2/cifar/train/"
            self.test_dir = "./../data/cifar2/cifar/test/"
            self.results_dir = "./../data/results/"
            self.model_dir = "./../data/models/"
            self.log_dir = "./../data/logs/"            
