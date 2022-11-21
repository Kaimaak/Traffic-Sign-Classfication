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

model = load_model('my_model.h5')

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
    if   classNum == 0:
        return 'speed limit 20 km/h'
    elif classNum == 1:
        return 'Speed Limit 30 km/h'
    elif classNum == 2:
        return 'Speed Limit 50 km/h'
    elif classNum == 3:
        return 'Speed Limit 60 km/h'
    elif classNum == 4:
        return 'Speed Limit 70 km/h'
    elif classNum == 5:
        return 'Speed Limit 80 km/h'
    elif classNum == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNum == 7:
        return 'Speed Limit 100 km/h'
    elif classNum == 8:
        return 'Speed Limit 120 km/h'
    elif classNum == 9:
        return 'No passing'
    elif classNum == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNum == 11:
        return 'Right-of-way at the next intersection'
    elif classNum == 12:
        return 'Priority road'
    elif classNum == 13:
        return 'Yield'
    elif classNum == 14:
        return 'Stop'
    elif classNum == 15:
        return 'No vechiles'
    elif classNum == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNum == 17:
        return 'No entry'
    elif classNum == 18:
        return 'General caution'
    elif classNum == 19:
        return 'Dangerous curve to the left'
    elif classNum == 20:
        return 'Dangerous curve to the right'
    elif classNum == 21:
        return 'Double curve'
    elif classNum == 22:
        return 'Bumpy road'
    elif classNum == 23:
        return 'Slippery road'
    elif classNum == 24:
        return 'Road narrows on the right'
    elif classNum == 25:
        return 'Road work'
    elif classNum == 26:
        return 'Traffic signals'
    elif classNum == 27:
        return 'Pedestrians'
    elif classNum == 28:
        return 'Children crossing'
    elif classNum == 29:
        return 'Bicycles crossing'
    elif classNum == 30:
        return 'Beware of ice/snow'
    elif classNum == 31:
        return 'Wild animals crossing'
    elif classNum == 32:
        return 'End of all speed and passing limits'
    elif classNum == 33:
        return 'Turn right ahead'
    elif classNum == 34:
        return 'Turn left ahead'
    elif classNum == 35:
        return 'Ahead only'
    elif classNum == 36:
        return 'Go straight or right'
    elif classNum == 37:
        return 'Go straight or left'
    elif classNum == 38:
        return 'Keep right'
    elif classNum == 39:
        return 'Keep left'
    elif classNum == 40:
        return 'Roundabout mandatory'
    elif classNum == 41:
        return 'End of no passing'
    elif classNum == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'

while True:
    # 이미지 읽어들이기
    suc, original = cap.read()

    # 이미지 처리
    img = np.asarray(original)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow('완료된 이미지', img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(original, "Class : ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(original, "Probability : ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # 예측 이미지
    prediction = model.predict(img)
    classIndex = np.argmax(model.predict(img), axis=-1)
    value = np.amax(prediction)
    if value > threshold:
        cv2.putText(original, str(classIndex) + " " + str(getClassname(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(original, str(round(value * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                    cv2.LINE_AA)
    cv2.imshow("Result", original)

    # q 키를 누른다면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
