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
path = "Datasets" # 데이터셋 경로
label = 'labels.csv' # 데이터 정보
batch_size = 50 # 필터 사이즈
per_epoch = 100 # 초당 학습 횟수
epoch = 20 # 최종 학습 횟수
imageDim = (32, 32, 3) # 이미지 차원
ratio = 0.2 # 비율
val_ratio = 0.2 # 검증 비율

# 이미지 임포트
cnt = 0
imgS = []
classNum = []
mylist = os.listdir(path)
numOfClass = len(mylist)
print("total classes :", len(mylist))

# 클래스 갯수 검사
for i in range(0, len(mylist)):
    myPick = os.listdir(path+"/"+str(cnt))
    for j in myPick:
        curImg = cv2.imread(path+"/"+str(cnt)+"/"+j)
        imgS.append(curImg)
        classNum.append(cnt)
    print(cnt, end=' ')
    cnt += 1
print(" ")

# 넘파이 배열로 변경
imgS = np.array(imgS)
classNum = np.array(classNum)

x_train, x_test, y_train, y_test = train_test_split(imgS, classNum, test_size=ratio)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio)

print("Data Shapes")
print("Train", end="")
print(x_train.shape, y_train.shape)
print("Validation", end="")
print(x_val.shape, y_val.shape)
print("Test", end="")
print(x_test.shape, y_test.shape)
assert(x_train.shape[0] == y_train.shape[0])
assert(x_val.shape[0] == y_val.shape[0])
assert(x_test.shape[0] == y_test.shape[0])
assert(x_train.shape[1:] == (imageDim))
assert(x_val.shape[1:] == (imageDim))
assert(x_test.shape[1:] == (imageDim))

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

print(numSamples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, numClass), numSamples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# 이미지 전처리
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


x_train = np.array(list(map(preprocessing, x_train)))
x_val = np.array(list(map(preprocessing, x_val)))
x_test = np.array(list(map(preprocessing, x_test)))
cv2.imshow("GrayScale Images", x_train[random.randint(0, len(x_train) - 1)])

# 1차원 추가하여 모양 변화 시키기
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# 이미지 데이터 증강
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(x_train)
batches = dataGen.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)

# 데이터 증강된 이미지 보여주기
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshape(imageDim[0], imageDim[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, numOfClass)
y_val = to_categorical(y_val, numOfClass)
y_test = to_categorical(y_test, numOfClass)


# CNN 모델 생성
def myModel():
    no_Of_Fileters = 60
    size_of_Filter = (5, 5)

    size_of_Filters2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    # 활성화 함수로 ReLU 사용
    model = Sequential()
    model.add((Conv2D(no_Of_Fileters, size_of_Filter, input_shape=(imageDim[0], imageDim[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_Fileters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Fileters // 2, size_of_Filters2, activation='relu')))
    model.add((Conv2D(no_Of_Fileters // 2, size_of_Filters2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numOfClass, activation='softmax'))

    # CNN 모델 컴파일
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 훈련
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=per_epoch, epochs=epoch, validation_data=(x_val, y_val),
                              shuffle=1)

# 시각화
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

model.save('my_model.h5')