# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:38:48 2018

@author: fd
"""

import cv2
import numpy as np
import os
import random
import copy
def detectFaces(img):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

def drawcat(location,cat):
    im=catlist[cat]
    H = np.float64([[1, 0, 0], [0, 1, location]])
    res = cv2.warpAffine(im, H, (w,h))  # 需要图像、变换矩阵、变换后的大小
    x=np.array(background,dtype="uint8")
    x[location:h,:,:]=res[location:h,:,:]
    return (x)


h=1080
w=1920
catlist=[]
for i in os.listdir("cat"):
    catlist.append(cv2.resize(cv2.imread("cat/"+i,-1),(w,h)))
background=np.ones((h,w,4),dtype="uint8")*255
cap = cv2.VideoCapture(0)
now_location=0
now_cat=0
while(1):
    # get a frame
    ret, frame = cap.read()
    cv2.imshow("cap", frame)
    # show a frame
    face_location=detectFaces(frame)
    Updatedraw = True
    if face_location==[]:#没有人脸
        dest_cat_location=0 #猫出现在最高
    else:
        dest_cat_location=h-face_location[0][1]
    if dest_cat_location>h-100:#换猫
        now_cat=random.randint(0, len(catlist)-1)
    if abs(now_location-dest_cat_location)<20 and dest_cat_location>0:
        Updatedraw=False
        cv2.imshow("capture", background)
    #如果猫应该显示那么绘制过渡动画 3帧
    '''while Updatedraw:
        for i in range(3):
            img=drawcat(int(now_location+(dest_cat_location-now_location)/3),now_cat)
            cv2.imshow("capture", img)
            now_location=dest_cat_location
        Updatedraw=False'''
    if Updatedraw:
        img = drawcat(dest_cat_location, now_cat)
        cv2.imshow("capture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
