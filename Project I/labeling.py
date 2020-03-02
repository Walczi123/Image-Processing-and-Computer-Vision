import numpy as np
import cv2 
from mask import mask

image_path = './multi_plant/rgb_00_03_000_00.png'

colors = [(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(128,128,128),(255,128,128),(128,255,128),(128,128,255)]    

# additional fucntion for overlapping mask
# remove the are intersection of mask1 and mask2 from mask1
def removeBitwiseMask(mask1, mask2): 
    m2 =  cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)
    ret,th2 = cv2.threshold(m2,0,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(th2)
    mask1 = cv2.bitwise_and(mask1 , mask1 , mask=mask_inv) 
    return mask1

# the main function
# finding the mask for leaves
def label(p,test):
    # read mask from mask fucntion
    my_mask = mask(p,False)
    if test :
        cv2.imshow("Mask", my_mask) 
    height, width = my_mask.shape

    # find the biggest contour
    ret,thresh = cv2.threshold(my_mask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    maxContour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    maxContourArea = cv2.contourArea(maxContour)

    #cases of erosion and dilatalion depends on the area
    if maxContourArea < 7500 :
        size = int(cv2.contourArea(maxContour)//10000) + 5 
        erod = size
        dilat = 3 * size
        minArea = 0
    else:
        if maxContourArea < 15000 :
            size = int(cv2.contourArea(maxContour)//5000) + 13
            erod = size
            dilat = 2 * size
            minArea = 0
        else:
            if maxContourArea < 35000:
                size = int(cv2.contourArea(maxContour)//5000) + 10
                erod = size
                dilat = 2.2 * size
                minArea = 500 
            else:
                size = int(cv2.contourArea(maxContour)//10000) + 3
                erod = size
                dilat = 4.1 * size
                minArea = 0
    dilat = int(dilat)

    # pefrome erosion
    mask_erode = cv2.erode(my_mask, None, iterations = erod)
    if test :
        cv2.imshow("Mask  Closed", mask_erode) 
        cv2.waitKey()

    # take all contour in the mask_erode
    ret,thresh = cv2.threshold(mask_erode,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    l = list()
    iter=0

    # sort the contours
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt)>minArea:
            # make the mask of the contour and dilate it
            m = cv2.drawContours(np.zeros((height ,width ,3), np.uint8 ), [cnt], 0, colors[iter%(len(colors)+1)], cv2.FILLED) 
            m = cv2.dilate(m, None, iterations = dilat)
            # segmented with my_mask
            m = cv2.bitwise_and(m, m, mask=my_mask)
            iter=iter+1
            # add to list
            l.append(m)

    # for every mask in list
    for i in range(0,len(l)-1):
        #from bigger mask remove the intersection with the smaller
        l[i] = removeBitwiseMask(l[i], l[i+1])
           
    if test :
        cv2.imshow("Sum(l)", sum(l)) 
        cv2.waitKey()   
    #return the list of mask
    return l

# cv2.imshow("Result",sum(label(image_path, True)))
# cv2.waitKey()
