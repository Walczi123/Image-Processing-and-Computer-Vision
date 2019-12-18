import cv2
import numpy as np


#cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

image = cv2.imread('./multi_plant/rgb_00_00_003_02.png',cv2.IMREAD_COLOR)

ddepth = cv2.CV_32F
gradX = cv2.Sobel(image, ddepth=ddepth, dx=1, dy=0, ksize=-9)
gradY = cv2.Sobel(image, ddepth=ddepth, dx=0, dy=1, ksize=-9) 
# cv2.imshow("gradX", gradX)
# cv2.imshow("gradY", gradY)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
# cv2.imshow("gradient", gradient)
gradient = cv2.convertScaleAbs(gradient)
# cv2.imshow("gradient_convert", gradient)
# cv2.waitKey()
   
Lower=[0, 0,0]
Upper=[179, 255, 255]



# Creating track bar
cv2.createTrackbar('h M', 'result',Upper[0],179,nothing)
cv2.createTrackbar('h L', 'result',Lower[0],179,nothing)
cv2.createTrackbar('s M', 'result',Upper[1],255,nothing)
cv2.createTrackbar('s L', 'result',Lower[1],255,nothing)
cv2.createTrackbar('v M', 'result',Upper[2],255,nothing)
cv2.createTrackbar('v L', 'result',Lower[2],255,nothing)
# cv2.createTrackbar('blur1', 'result',1,20,nothing)
# cv2.createTrackbar('blur2', 'result',1,20,nothing)


while(1):

    #_, frame = cap.read()

    #converting to HSV
    hsv = cv2.cvtColor(gradient,cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    hM = cv2.getTrackbarPos('h M','result')
    hL = cv2.getTrackbarPos('h L','result')
    sM = cv2.getTrackbarPos('s M','result')
    sL = cv2.getTrackbarPos('s L','result')
    vM = cv2.getTrackbarPos('v M','result')
    vL = cv2.getTrackbarPos('v L','result')

    # Normal masking algorithm
    lower_blue = np.array([hL,sL,vL])
    upper_blue = np.array([hM,sM,vM])

    res = cv2.inRange(hsv,lower_blue, upper_blue)
    result = cv2.bitwise_and(image,image,mask = res)

    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print([hL,sL,vL])
        print([hM,sM,vM])
        break

#cap.release()

cv2.destroyAllWindows()