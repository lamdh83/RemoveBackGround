import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture('1.avi')
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread('Images/1.jpg')
imgBg = cv2.resize(imgBg,(640, 480))
imgIndex = 0

listImg = os.listdir('Images')
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    img = cv2.resize(img,(640, 480))
    imgList.append(img)


while True:
    success, img = cap.read()
    # imgOut = segmentor.removeBG(img, (0, 255, 0), threshold=0.8)
    imgOut = segmentor.removeBG(img, imgList[imgIndex], threshold=0.8)

    imgStacked = cvzone.stackImages([img, imgOut],2,0.8)
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))
    cv2.imshow('image out', imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if imgIndex > 0:
            imgIndex -=1
    elif key == ord('d'):
        if imgIndex < len(imgList) - 1:
            imgIndex +=1
    elif key == ord('q'):
        break

