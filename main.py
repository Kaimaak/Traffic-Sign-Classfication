import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# 파라미터
path = "Datasets"
label = 'labels.csv'
batch_size = 50
per_epoch = 100
epoch = 10
imageDim = (32, 32, 3)
ratio = 0.2
val_ratio = 0.2

# 이미지 임포트
cnt = 0
imgS = []
classNum = []
mylist = os.listdir(path)
numOfClass = len(mylist)
print("total classes :", len(mylist))

for i in range(0, len(mylist) - 1):
    myPick = os.listdir(path+"/"+str(cnt))
    for j in myPick:
        curImg = cv2.imread(path+"/"+str(cnt)+"/"+j)
        imgS.append(curImg)
        classNum.append(cnt)
    print(cnt, end=' ')
    cnt += 1
print(" ")

imgS = np.array(imgS)
classNum = np.array(classNum)

x_train, x_test, y_train, y_test = train_test_split(imgS, classNum, test_size=ratio)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio)

print("Data Shapes")
print("Train", end="");print(x_train.shape, y_train.shape)
print("Validation", end="");print(x_val.shape, y_val.shape)
print("Test", end="");print(x_test.shape, y_test.shape)
assert(x_train.shape[0] == y_train.shape[0])
assert(x_val.shape[0] == y_val.shape[0])
assert(x_test.shape[0] == y_test.shape[0])
assert(x_train.shape[1:] == imageDim)
assert(x_val.shape[1:] == imageDim)
assert(x_test.shape[1:] == imageDim)

data = pd.read_csv(label)
print("data shape", data.shape, type(data))

numSamples = []
cols = 5
numClass = numOfClass
fig, axs = plt.subplots(nrows=numClass, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+"-"+row["Name"])
            numSamples.append(len(x_selected))

