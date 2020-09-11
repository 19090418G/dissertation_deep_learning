import cv2, os, math, random
import numpy as np
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, imgPath, batch_size=16, target_size=(224, 224)):
        self.imgPath_list = np.array(imgPath)
        self.batch_size = batch_size
        self.target_size = target_size
        self.indexes = np.arange(len(self.imgPath_list))

    def __len__(self):
        return int(np.floor(len(self.imgPath_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x, y = [], []
        for i in self.imgPath_list[indexes]:
            img = cv2.imread('img\{}'.format(i))
            img = cv2.resize(img, (224, 224))
            img = np.array(img, dtype=np.float)
            x.append(img)
            y.append(int(i.split('.')[0].split('_')[1]))

        x, y = np.array(x, dtype=np.float) / 255.0, to_categorical(y, num_classes=2)

        return x, y