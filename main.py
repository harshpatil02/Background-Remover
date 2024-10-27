import cv2
import cvzone
from cvzone.FPS import FPS
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = FPS()

listimg= os.listdir("Images")
imglist=[]
for imgpath in listimg:
    img =cv2.imread(f'Images/{imgpath}')
    imglist.append(img)

indeximg=0
while True:
    success, img = cap.read()
    imgout = segmentor.removeBG(img, imglist[indeximg], cutThreshold=0.8)

    imgstack = cvzone.stackImages([img, imgout], 2, 1)
    _,imgstack = fpsReader.update(imgstack,bgColor=(0,0,255))
    cv2.imshow("Image", imgstack)

    # cv2.imshow("Image out",imgOut)
    key=cv2.waitKey(1)
    if key==ord('a'):
        if indeximg>0:
            indeximg-=1
    elif key ==ord('d'):
        if indeximg<len(imglist)-1:
            indeximg+=1
    elif key==ord('q'):
        break
    cv2.waitKey(1)
