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
    if classNum == 0:
        return '20 km/h 제한'
    elif classNum == 1:
        return '30 km/h 제한'
    elif classNum == 2:
        return '50 km/h 제한'
    elif classNum == 3:
        return '60 km/h 제한'
    elif classNum == 4:
        return '70 km/h 제한'
    elif classNum == 5:
        return '80 km/h 제한'
    elif classNum == 6:
        return '제한 속도 80 km/h 종료'
    elif classNum == 7:
        return '100 km/h 제한'
    elif classNum == 8:
        return '120 km/h 제한'
    elif classNum == 9:
        return '추월 금지'
    elif classNum == 10:
        return '3.5톤 이상 차량 통행 금지'
    elif classNum == 11:
        return '다음 교차로에서 선로 진입'
    elif classNum == 12:
        return '우선 순위 도로'
    elif classNum == 13:
        return '양보하시오'
    elif classNum == 14:
        return '멈추시오'
    elif classNum == 15:
        return '차량 금지'
    elif classNum == 16:
        return '3.5톤 이상 차량 금지'
    elif classNum == 17:
        return '들어오지 마시오'
    elif classNum == 18:
        return '주의'
    elif classNum == 19:
        return '왼쪽에 위험한 커브길'
    elif classNum == 20:
        return '오른쪽에 위험한 커브길'
    elif classNum == 21:
        return '이중 곡선'
    elif classNum == 22:
        return '울퉁불퉁한 도로'
    elif classNum == 23:
        return '미끄럼 주의'
    elif classNum == 24:
        return '오른쪽 길이 좁아'
    elif classNum == 25:
        return '도로 공사중'
    elif classNum == 26:
        return '교통 신호등'
    elif classNum == 27:
        return '보행자'
    elif classNum == 28:
        return '어린이 횡단보도'
    elif classNum == 29:
        return '자전거 횡단보도'
    elif classNum == 30:
        return '얼음/눈 주의'
    elif classNum == 31:
        return '야생동물이 건너는 도로'
    elif classNum == 32:
        return '아우토반'
    elif classNum == 33:
        return '앞으로 우회전'
    elif classNum == 34:
        return '앞으로 좌회전'
    elif classNum == 35:
        return '계속 직진'
    elif classNum == 36:
        return '직진하거나 오른쪽으로'
    elif classNum == 37:
        return '직진하거나 왼쪽으로'
    elif classNum == 38:
        return '우측 통행'
    elif classNum == 39:
        return '좌측 통행'
    elif classNum == 40:
        return '우회 필수'
    elif classNum == 41:
        return '통과 영역 없음'
    elif classNum == 42:
        return '3.5톤 이상의 차량 통행 금지 종료'

while True:
    # 이미지 읽어들이기
    suc, original = cap.read()

    # 이미지 처리
    img = np.asarray(original)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow('완료된 이미지', img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(original, "클래스 : ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(original, "확률 : ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # 예측 이미지
    prediction = model.predict(img)
    classIndex = np.argmax(model.predict(img), axis=-1)
    value = np.amax(prediction)
    