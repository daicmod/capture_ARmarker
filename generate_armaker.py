import cv2.aruco
import numpy as np

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
SIZE = 150
ImgList1 = []
ImgList2 = []
WiteList = []


def arGenerator():
    img_white = np.ones((SIZE, SIZE, 3), np.uint8)*255
    for i in range(1, 6):
        fileName = "ar_" + str(i) + ".png"
        generator = aruco.drawMarker(dictionary, i, SIZE)
        cv2.imwrite(fileName, generator)
        ImgList1.append(cv2.imread(fileName))
        ImgList1.append(img_white)
        WiteList.append(img_white)
        WiteList.append(img_white)
        convImg1 = cv2.hconcat(ImgList1)
        convWhite = cv2.hconcat(WiteList)
    for i in range(6, 11):
        fileName = "ar_" + str(i) + ".png"
        generator = aruco.drawMarker(dictionary, i, SIZE)
        cv2.imwrite(fileName, generator)
        ImgList2.append(cv2.imread(fileName))
        ImgList2.append(img_white)
        convImg2 = cv2.hconcat(ImgList2)
    TestList = [convImg1, convWhite, convImg2]
    convImg3 = cv2.vconcat(TestList)
    cv2.imshow('ArMarker1', convImg3)
    cv2.imwrite("Result.jpg", convImg3)
    cv2.waitKey(0)


arGenerator()
