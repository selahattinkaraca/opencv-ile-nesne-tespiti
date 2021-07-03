import cv2
import numpy as np
img=cv2.imread("resimler/van.png", cv2.IMREAD_COLOR)
frameWidth = 230
frameHeight = 250
def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        hesaplanan_area = cv2.contourArea(cnt)
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)


imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
threshold1 =200
threshold2 = 130
imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
cekirdek = np.ones((5, 5))
imgDil = cv2.dilate(imgCanny, cekirdek, iterations=1)
getContours(imgDil,imgContour)
cv2.imshow("img", imgContour)
cv2.waitKey(0)
