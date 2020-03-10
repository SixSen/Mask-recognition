import cv2
import time
import numpy as np
import pygame
import threading
import os

file_slogan = r'radio/slogan.mp3'
file_slogan_short = r'radio/slogan_short.mp3'
pygame.mixer.init(frequency=16000, size=-16, channels=2, buffer=4096)


def nothing(x):  # 滑动条的回调函数
    pass


# 播放请带好口罩音频
def slogan_short():
    timeplay = 1.5
    global playflag_short
    playflag_short = 0
    while True:
        if playflag_short == 1:
            track = pygame.mixer.music.load(file_slogan_short)
            print("------------请您戴好口罩")
            pygame.mixer.music.play()
            time.sleep(timeplay)
            playflag_short = 0
        time.sleep(0)
        # print("slogan_shorttread running")


# 播放宣传词语
def slogan():
    timeplay = 18
    global playflag
    playflag = 0
    while True:
        if playflag == 1:
            track = pygame.mixer.music.load(file_slogan)
            print("------------全国疾控中心提醒您：预防千万条，口罩第一条。")
            pygame.mixer.music.play()
            time.sleep(timeplay)
            playflag = 0
        time.sleep(0)
        # print("slogantread running")


# 霍夫变换找直线
def line_detect_possible_demo(image):
    # 通常用第二种方式。
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray1, 50, 150, apertureSize=3)
    # 自动检测可能的直线，返回的是一条条线段
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
    print(type(lines))
    # 在image上标出所有直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (200, 0, 0), 2)
    cv2.imshow("linepossible_demo", image)


# 面部识别初始化
def facesdetecter_init():
    global thread_slogan
    global thread_slogan_short

    thread_slogan = threading.Thread(target=slogan).start()
    thread_slogan_short = threading.Thread(target=slogan_short).start()
    # 导入背景
    image = cv2.imread("images/backgound.jpg")
    cv2.imshow('skin', image)
    cv2.createTrackbar("minH", "skin", 10, 180, nothing)
    cv2.createTrackbar("maxH", "skin", 50, 180, nothing)


# 面部识别
def facesdetecter(image):
    start = time.time()  # 开始时间，用于计算帧率
    timer = cv2.getTickCount()
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 将图片高斯模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度
    image2 = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将图片转化成HSV格式
    cv2.imshow("hsv", hsv)  # 显示HSV图
    H, S, V = cv2.split(hsv)
    # cv2.imshow("hsv-H",H)#显示HSV图明度
    # ret, thresh =cv2.threshold(H,17,25,0)#阈值分割
    minH = cv2.getTrackbarPos("minH", 'skin')
    maxH = cv2.getTrackbarPos("maxH", 'skin')
    if minH > maxH:
        maxH = minH
    # print("maxh=",maxH)
    # thresh_h=cv2.inRange(H,30/2,50/2)#0-180du 提取
    thresh_h = cv2.inRange(H, minH, maxH)  # 0-180du 提取人体肤色区域
    # ret,thresh_h=cv2.threshold(H, 30/2, 50/2, cv2.THRESH_BINARY)#二值化
    # ret,thresh_h=cv2.threshold(thresh_h, 155, 255, cv2.THRESH_BINARY)#二值化
    cv2.imshow("skin", thresh_h)  # 显示二值化图

    # dilateh = cv2.dilate(thresh_h, None, iterations=2)#膨胀
    # cv2.imshow('dilateh', dilateh)
    # erode = cv2.erode(dilate, None, iterations=2)# 腐蚀
    # cv2.imshow('erodeh', erode)

    canny = cv2.Canny(gray, 50, 150)
    # cv2.imshow("canny",canny)#显示边缘处理图

    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)#人脸检测
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)  # 眼睛检测
    # upperbody = upperbody_cascade.detectMultiScale(gray, 1.3, 5)#上身检测
    # mouth=mouth_cascade.detectMultiScale(gray, 1.3, 5)#嘴巴检测
    # nose=nose_cascade.detectMultiScale(gray, 1.3, 5)#鼻子检测
    # lefteye=lefteye_cascade.detectMultiScale(gray, 1.3, 5)#左眼检测
    # for (x,y,w,h) in faces:
    # frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    # 标出眼睛的位置
    for (x, y, w, h) in eyes:
        frame = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色方框
        # frame = cv2.rectangle(image,(x+10,y+10),(x+w,y+h),(0,255,0),2)
        # print(eyes)
        # print("--eyes------")
    # for (x,y,w,h) in upperbody:
    #     frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    # for (x,y,w,h) in mouth:
    #     frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    # for (x,y,w,h) in nose:
    # frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    # for (x,y,w,h) in lefteye:
    # frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,250),2)
    # 计算帧率
    total_area_mask = 0
    total_area_eyes = 0
    rect_eyes = []
    if len(eyes) > 1:
        (x1, y1, w1, h1) = eyes[0]
        for (x, y, w, h) in eyes[1:]:
            (x2, y2, w2, h2) = (x, y, w, h)
            # print(len(eyes)%2)
            # print(x2,y2,w2,h2)
            # if len(eyes)%2==1 and (x2,y2,w2,h2)==eyes[-1]:
            #    continue
            rect_eyes.append((x1, y1, x2 + w2 - x1, y2 + h2 - y1))
            (x1, y1, w1, h1) = (x2, y2, w2, h2)

        # 计算眼睛部分的肤色面积
        for (x, y, w, h) in rect_eyes:
            frame = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 250, 255), 2)
            thresh_eyes = thresh_h[y:y + h, x:x + w]
            contours, hierarchy = cv2.findContours(thresh_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景
            cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算眼睛部分的面积
                total_area_eyes += Area
        if total_area_eyes != 0:
            print("total_area_eyes=", total_area_eyes)
            frame = cv2.putText(image, "Eyes Area : {:.3f}".format(total_area_eyes), (120, 40),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 255, 0), 1)  # 绘制

        # 口罩区域
        rect_mask = [(x, y + h, w, h * 2)]  # 口罩区域是眼睛的区域的下方的两倍高度位置

        # 口罩区域
        for (x, y, w, h) in rect_mask:
            frame = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            thresh_mask = thresh_h[y:y + h, x:x + w]
            # image2[y:y+h,x:x+w]=thresh_h
            contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景
            cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算口罩区域面积
                total_area_mask += Area
        if total_area_eyes != 0:
            print("total_area_mask=", total_area_mask)
            frame = cv2.putText(image, "Mask Area : {:.1f}".format(total_area_mask), (120, 80), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 255), 1)  # 显示口罩区域面积

        # print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积
        if total_area_eyes == total_area_mask == 0:
            print("正在检测...")
            frame = cv2.putText(image, "Scanning", (rect_eyes[0][0], rect_eyes[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 100, 100), 1)  # 显示正在扫描（眼睛与口罩面积均不存在）

        if total_area_eyes < total_area_mask:
            print("----------------没有口罩")
            global playflag_short
            playflag_short = 1
            frame = cv2.putText(image, "No Mask", (rect_eyes[0][0], rect_eyes[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 255), 1)  # 显示没有佩戴口罩（红色字）

        if total_area_eyes > total_area_mask:
            global thread_slogan
            print("----------------------戴口罩了")
            global playflag
            playflag = 1
            frame = cv2.putText(image, "Have Mask", (rect_eyes[0][0], rect_eyes[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 255, 0), 1)  # 显示佩戴口罩了（绿色字）

            # thread = threading.Thread(target=playslogan,args=(2)).start()
            # cv2.imshow("hsv-H-threshold-roi",thresh_h)#显示二值化图
    end = time.time()  # 结束时间
    fps = 1 / (end - start)  # 帧率
    # Calculate Frames per second (FPS)
    # fps1 = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    frame = cv2.putText(image, "fps:{:.3f}".format(fps), (550, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)  # 绘制
    # frame = cv2.putText(image,"fps:{:.3f}".format(fps1),(300,25),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)#绘制

    # cv2.imshow("face_f",image2)#显示肤色检测图片
    cv2.imshow("face", image)  # 显示最终图片
    # line_detect_possible_demo(image)#霍夫变换找直线


def mogseparate(image):
    fgmask = mog.apply(image)
    ret, binary = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow("fgmask", fgmask)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    backgimage = mog.getBackgroundImage()
    # 查找轮廓
    # contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(binary,contours,-1,(0,0,255),3)
    binary = cv2.erode(binary, None, iterations=4)  # 腐蚀
    cv2.imshow('erode', binary)
    cv2.imshow("backgimage", backgimage)
    cv2.imshow("frame", image)
    cv2.imshow("binary", binary)


def knnseperate(image):
    mog_sub_mask = mog2_sub.apply(image)
    knn_sub_mask = knn_sub.apply(image)

    cv2.imshow('original', image)
    cv2.imshow('MOG2', mog_sub_mask)
    cv2.imshow('KNN', knn_sub_mask)


def trackAvg(image, k_write):
    if k_write == 1:
        cv2.imwrite("images/backgound.jpg", image)
        k_write = 0
    background = cv2.imread("images/backgound.jpg")
    # cv2.RunningAvg(image, background, 0.1, None)
    # cv2.imshow('live', frame)
    # cv2.imshow('avg',avg_show)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")
'''此文件是opencv的haar人脸特征分类器'''
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
eyes_cascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
'''此文件是opencv的haar眼镜特征分类器'''

upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
upperbody_cascade.load("data/haarcascades/haarcascade_upperbody.xml")
'''此文件是opencv的haar上半身特征分类器'''

mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
mouth_cascade.load("data/haarcascades/haarcascade_mcs_mouth.xml")
'''此文件是opencv的haar上半身特征分类器'''

nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
nose_cascade.load("data/haarcascades/haarcascade_mcs_nose.xml")
'''此文件是opencv的haar上半身特征分类器'''

lefteye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
lefteye_cascade.load("data/haarcascades/haarcascade_lefteye_2splits.xml")
'''此文件是opencv的haar上半身特征分类器'''

mog = cv2.createBackgroundSubtractorMOG2()
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

knn_sub = cv2.createBackgroundSubtractorKNN()
mog2_sub = cv2.createBackgroundSubtractorMOG2()

if __name__ == '__main__':

    facesdetecter_init()
    k_write = 1
    capture = cv2.VideoCapture(0)
    image = cv2.imread("images/4.jpg")
    # while 0:
    #     facesdetecter(image)  # 对图片检测
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    while True:

        ref, frame = capture.read()
        if ref == False:
            print("打开摄像头错误")
            break
        # cv2.imshow("frame",frame)
        # 等待30ms显示图像
        c = cv2.waitKey(30) & 0xff
        if c == 27:  # ESC退出
            capture.release()
            break
        facesdetecter(frame)  # 对视频检测

        # mogseparate(frame)#mog方式分离前景
        knnseperate(frame)#knn方式分离前景
        # trackAvg(frame,k_write)#runningavg分离前景

    # cap.release()
    cv2.destroyAllWindows()
    print("byebye~")
    os._exit(0)
