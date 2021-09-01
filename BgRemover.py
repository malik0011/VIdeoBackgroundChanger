import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread("wallpapersden.com_planet-rising-over-galaxy_640x480.jpg")
while True:
    success, img =cap.read()
    imgOut = segmentor.removeBG(img,imgBg,threshold=0.50)


    imgstack = cvzone.stackImages([img,imgOut],2,1)
    _, imgstack = fpsReader.update(imgstack,color=(255,255,255))
    cv2.imshow("Imagestack",imgstack)
    cv2.imshow("ImageOut", imgOut)
    cv2.waitKey(1)