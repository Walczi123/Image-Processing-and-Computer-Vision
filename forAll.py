import glob
import numpy as np
import cv2 

path = './multi_plant'


def code(p):
    print(p)
    image = cv2.imread(p,cv2.IMREAD_COLOR)  
    height, width, channels = image.shape
   
    # remove all colors except of particular shades of green 
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV) 
    lower_green = np.array([19, 31, 33],np.uint8) 
    upper_green = np.array([86, 255, 134],np.uint8)
    mask = cv2.inRange(hsv, lower_green , upper_green)

    # blur , optionally use opening if USEOPENING: 
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN , kernel) 
    mask = cv2.medianBlur(mask , 3)

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

    mask = cv2.drawContours(np.zeros((height ,width ,3), np.uint8 ), [cnt], 0, (255,255,255), cv2.FILLED) 
    mask = cv2.cvtColor(mask ,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image", mask)

    #cv2.waitKey()

    segmented = cv2.bitwise_and(image , image , mask=mask) 
    # cv2.imshow("seg", segmented)
    # cv2.waitKey()

    hsv = cv2.cvtColor(segmented , cv2.COLOR_BGR2HSV) 
    lower_brown = np.array([38, 31, 33],np.uint8) 
    upper_brown = np.array([86, 255, 134],np.uint8)
    mask_seg = cv2.inRange(hsv, lower_brown , upper_brown)
    kernel22 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    mask_seg = cv2.morphologyEx(mask_seg, cv2.MORPH_OPEN , kernel22)
    # cv2.imshow("Mask Seg", mask_seg)
    # cv2.waitKey()

    # perform a series of erosions and dilations
    mask_seg = cv2.erode(mask_seg, None, iterations = 1)
    mask_seg = cv2.dilate(mask_seg, None, iterations = 1)
    cv2.imshow("Mask  Closed", mask_seg)


    # ret2,thresh2 = cv2.threshold(mask_seg,127,255,0)
    # contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # c2 = sorted(contours2, key = cv2.contourArea, reverse = True)[0]
    # cv2.drawContours(image, [c2], 0, (0,0,0), 3)
    # cv2.imshow("Show end ",image)


    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Show",image)
    cv2.waitKey()
    

files = [f for f in glob.glob(path + "**/*.png", recursive=True)]
for f in files:
    #print(f)
    code(f)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()