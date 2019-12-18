import numpy as np
import cv2
import cvutils

image = cv2.imread('./multi_plant/rgb_00_00_003_02.png',cv2.IMREAD_COLOR)

ddepth = cv2.CV_32F
gradX = cv2.Sobel(image, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(image, ddepth=ddepth, dx=0, dy=1, ksize=-1) 
cv2.imshow("gradX", gradX)
cv2.imshow("gradY", gradY)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
cv2.imshow("gradient", gradient)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("gradient_convert", gradient)

cv2.waitKey()