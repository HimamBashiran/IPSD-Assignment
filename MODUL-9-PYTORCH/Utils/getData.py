############################# GETDATA.PY #######################################################
import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, folder="/Users/User/IPSD Assignment/MODUL-9-PYTORCH/Dataset/"):
        self.dataset = []
        onehot = np.eye(5)
        for _, i in enumerate(os.listdir(folder)):
            for j in os.listdir(folder + i):
                image = cv.resize(cv.imread(folder + i + "/" + j), (200, 200))/255
                self.dataset.append([image, onehot[_]])
        print(self.dataset)

    def __len__(self):
        """
            :return: panjang list dataset
        """
        return len(self.dataset)

    def __getitem__(self, item):
        """
        param item: index
        :return: mengembalikan item dataset
        """
        features, label = self.dataset[item]
        return (torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
    
if __name__=="__main__":
    data = Data()