class Properties:
    def __init__(self, environment):
        if (environment == "colab"):
            self.data_dir = "/content/drive/My Drive/DL-project/data/beach/"
            self.test_dir = "/content/drive/My Drive/DL-project/data/test/"
            self.results_dir = "/content/drive/My Drive/DL-project/data/gan/beach/results/"
            self.model_dir = "/content/drive/My Drive/DL-project/data/gan/beach/models/"
            self.log_dir = "/content/drive/My Drive/DL-project/data/gan/beach/logs/"
        else :
            #self.data_dir = "./../data/beach"
            #self.test_dir = "./../data/test"
            self.data_dir = "./../data/beach/train/"
            self.test_dir = "./../data/beach/test/"
            self.results_dir = "./../data/beach/results/"
            self.model_dir = "./../data/beach/models/"
            self.log_dir = "./../data/beach/logs/"            
