import cv2
import numpy as np


#cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

image = cv2.imread('./multi_plant/rgb_01_02_008_00.png',cv2.IMREAD_COLOR)
height, width, channels = image.shape

   
# remove all colors except of particular shades of green 
hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
lower_green = np.array([31, 28, 51],np.uint8) #[30, 22, 22]  40 55 40
upper_green = np.array([73, 104, 152],np.uint8) #[85, 235, 195] 120 130 110
mask = cv2.inRange(hsv, lower_green , upper_green)


Lower=[37, 31, 33]
Upper=[86, 255, 134]



# Creating track bar
cv2.createTrackbar('h M', 'result',Upper[0],179,nothing)
cv2.createTrackbar('h L', 'result',Lower[0],179,nothing)
cv2.createTrackbar('s M', 'result',Upper[1],255,nothing)
cv2.createTrackbar('s L', 'result',Lower[1],255,nothing)
cv2.createTrackbar('v M', 'result',Upper[2],255,nothing)
cv2.createTrackbar('v L', 'result',Lower[2],255,nothing)
cv2.createTrackbar('blur1', 'result',1,20,nothing)
cv2.createTrackbar('blur2', 'result',1,20,nothing)


while(1):

    #_, frame = cap.read()

    #converting to HSV
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    hM = cv2.getTrackbarPos('h M','result')
    hL = cv2.getTrackbarPos('h L','result')
    sM = cv2.getTrackbarPos('s M','result')
    sL = cv2.getTrackbarPos('s L','result')
    vM = cv2.getTrackbarPos('v M','result')
    vL = cv2.getTrackbarPos('v L','result')
    b1 = cv2.getTrackbarPos('blur1','result')
    b2 = cv2.getTrackbarPos('blur2','result')

    # Normal masking algorithm
    lower_blue = np.array([hL,sL,vL])
    upper_blue = np.array([hM,sM,vM])

    res = cv2.inRange(hsv,lower_blue, upper_blue)
    kernel123 = cv2.getStructuringElement(cv2.MORPH_RECT, (b1,b2))
    res = cv2.morphologyEx(res , cv2.MORPH_OPEN , kernel123) 
    #res = cv2.medianBlur(res , b)
    result = cv2.bitwise_and(image,image,mask = res)

    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print([hL,sL,vL])
        print([hM,sM,vM])
        break

#cap.release()

cv2.destroyAllWindows()