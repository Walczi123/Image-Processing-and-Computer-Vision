import glob
import numpy as np
import cv2 
import os
from main_code import main

pathToRead = './multi_plant'
pathToWrite = './'
directoryName = 'my_bounding_boxes'
directory = os.path.dirname(pathToWrite+directoryName)

def bounding_boxes(p):
    mask = main(p,False)
    image = cv2.imread(p,cv2.IMREAD_COLOR)
    x,y,w,h = cv2.boundingRect(mask)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    return image

def save_image(name, image):
    name = os.path.basename(name)
    cv2.imwrite(pathToWrite+directoryName+'/'+name, image)

files = [f for f in glob.glob(pathToRead + "**/*.png", recursive=True)]
if not os.path.exists(directory):
    os.makedirs(directory)
i=0
try :
    for f in files:    
        image = bounding_boxes(f)
        save_image(f, image)
        print(i/9)
        i+=1
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
except Exception as e:
        print(e)
        print(f)    