# import the necessary packages
import numpy as np
import cv2 
import random
import os
from math import sqrt as sqrt

def mask(path, test):
    # load the image
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    if test :
        cv2.imshow("Image", image) 
        cv2.waitKey()
    
    # remove generally all colors exept shades of green
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV) 
    lower_green = np.array([0, 0, 0],np.uint8) 
    upper_green = np.array([179, 255, 165],np.uint8)
    mask = cv2.inRange(hsv, lower_green , upper_green)
    if test :
        cv2.imshow("Image", mask)
        cv2.waitKey()

    # finding contur of the biggest area
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    # if the biggest contur has the area > 40000 then remove more precisely all colors except green
    if cv2.contourArea(c) > 40000:
        segmented = cv2.bitwise_and(image , image , mask=mask)
        hsv = cv2.cvtColor(segmented , cv2.COLOR_BGR2HSV) 
        lower_green = np.array([27, 29, 0],np.uint8) 
        upper_green = np.array([179, 255, 165],np.uint8)
        mask = cv2.inRange(hsv, lower_green , upper_green)
        if test :
            cv2.imshow("warunek", mask)
            cv2.waitKey()
        ret,thresh = cv2.threshold(mask,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # set minArea = 5000
        minArea = 5000
    else :
        # set minArea = 1000
        minArea = 1000

    # finding the generall shape of plant as the most center contour
    # of the area > minArea
    closest=1000
    for c in contours :
        if cv2.contourArea(c)>minArea:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dist=abs(width/2-cX)+abs(height/2-cY)
            if closest > dist: 
                closest=dist
                cnt=c

    # take the general mask of the plant
    mask = cv2.drawContours(np.zeros((height ,width ,3), np.uint8 ), [cnt], 0, (255,255,255), cv2.FILLED) 
    mask = cv2.cvtColor(mask ,cv2.COLOR_BGR2GRAY)
    if test :
        cv2.imshow("Image", mask)
        cv2.waitKey()

    # take only part of the mask from the image
    segmented = cv2.bitwise_and(image , image , mask=mask) 
    if test : 
        cv2.imshow("seg", segmented)
        cv2.waitKey()
    
    # more precisely remove all colors exept green
    # only from the segmented part of the image
    hsv = cv2.cvtColor(segmented , cv2.COLOR_BGR2HSV) 
    lower_brown = np.array([37, 31, 28],np.uint8) 
    upper_brown = np.array([86, 255, 134],np.uint8)
    mask_seg = cv2.inRange(hsv, lower_brown , upper_brown)
    if test :
        cv2.imshow("Mask Seg", mask_seg)
        cv2.waitKey()

    # perform a series of erosions and dilations depends on the max contour area
    kernel22 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    mask_seg = cv2.morphologyEx(mask_seg, cv2.MORPH_OPEN , kernel22)
    ret,thresh = cv2.threshold(mask_seg,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # take the biggest contour
    maxContour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    # set size of the contour devided by 25000
    size = int(cv2.contourArea(maxContour)//25000)
    # perform erosion and dilatation with size iteration
    mask_seg = cv2.erode(mask_seg, None, iterations = size)
    mask_seg = cv2.dilate(mask_seg, None, iterations = size)
    if test :
        cv2.imshow("Mask  Closed", mask_seg) 
        cv2.waitKey()
    # and return the mask of the plant
    return mask_seg

# Debuging
# image_path = './multi_plant/rgb_00_01_000_04.png'
# cv2.imshow("Result",mask(image_path, True))
# cv2.waitKey()