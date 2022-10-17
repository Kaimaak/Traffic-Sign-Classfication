import cv2
import numpy as np
from keras.models import load_model

width = 640 # 가로
height = 480 # 세로
brightness = 180 # 밝기
threshold = 0.75 # 한계점
font = cv2.FONT_HERSHEY_SIMPLEX

# 비디오카메라 (웹캠)
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10, brightness)

model = load_model('model.h5')

def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    return img

def preprocessing(img):
    img = gray(img)
    img = equalize(img)
    img = img/225
    return img

def getClassname(classNum):
