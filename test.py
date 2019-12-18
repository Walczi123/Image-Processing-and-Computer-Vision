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
height, width, channels = image.shape

   
# remove all colors except of particular shades of green 
hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
lower_green = np.array([31, 28, 51],np.uint8) #[30, 22, 22]  40 55 40
upper_green = np.array([73, 104, 152],np.uint8) #[85, 235, 195] 120 130 110
mask = cv2.inRange(hsv, lower_green , upper_green)

# blur , optionally use opening if USEOPENING: 
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
# mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN , kernel) 
mask = cv2.medianBlur(mask , 3)

#cv2.imshow("Image", mask)
#cv2.waitKey(0)


ret,thresh = cv2.threshold(mask,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

closest=1000
for c in contours :
    if cv2.contourArea(c)>500:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        dist=abs(width/2-cX)+abs(height/2-cY)
        if closest > dist: 
            closest=dist
            cnt=c


x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.drawContours(mask , [cnt], 0, (0,255,255), 3)
#cv2.imshow("Show",image)

mask = cv2.drawContours(np.zeros((height ,width ,3), np.uint8 ), [cnt], 0, (255,255,255), cv2.FILLED) 
mask = cv2.cvtColor(mask ,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image", mask)
   
segmented = cv2.bitwise_and(image , image , mask=mask) 
cv2.imshow("seg", segmented)

hsv = cv2.cvtColor(segmented , cv2.COLOR_BGR2HSV) 
Lower=[37, 31, 33]
Upper=[86, 255, 134]
lower_brown = np.array([37, 31, 33],np.uint8) 
upper_brown = np.array([86, 255, 134],np.uint8)

mask_seg = cv2.inRange(hsv, lower_brown , upper_brown)
kernel12 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
mask_seg1 = cv2.morphologyEx(mask_seg , cv2.MORPH_OPEN , kernel12) 
cv2.imshow("Mask Seg", mask_seg1)


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
    hsv = cv2.cvtColor(segmented,cv2.COLOR_BGR2HSV)

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
    result = cv2.bitwise_and(segmented,segmented,mask = res)

    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print([hL,sL,vL])
        print([hM,sM,vM])
        break

#cap.release()

cv2.destroyAllWindows()