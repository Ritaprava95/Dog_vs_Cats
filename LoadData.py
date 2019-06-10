import os
from PIL import Image
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array

def load_data():
    folder = 'train/train/'
    train_X = list()
    for file in os.listdir(folder):
        img = Image.open(folder+file)
        img = np.array(img.resize((100,100)))
        train_X.append(img)
        
    train_X = np.array(train_X)
    np.save('train_X.npy', train_X)
    train_y = np.array([0 for x in range(12500)] + [1 for x in range(12500)])
    np.save('train_y.npy', train_y)
    return train_X, train_y

def load_data2():
    folder = 'train/train/'
    train_X = list()
    for file in os.listdir(folder):
        img = load_img(folder + file, target_size=(100, 100))
        img = img_to_array(img)
        train_X.append(img)
        
    train_X = np.asarray(train_X)
    train_y = np.array([0 for x in range(12500)] + [1 for x in range(12500)])
    np.save('train_X.npy', train_X)
    np.save('train_y.npy', train_y)
    return train_X, train_y


train_X, train_y = load_data2()

print(train_X.shape, train_y.shape)
