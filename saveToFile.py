import glob
import numpy as np
import cv2 
import os
from main_code import main

pathToRead = './multi_plant'
pathToWrite = './'
directoryName = 'my_plants'
directory = os.path.dirname(pathToWrite+directoryName)

def save_image(name, image):
    name = os.path.basename(name)
    cv2.imwrite(pathToWrite+directoryName+'/'+name, image)

files = [f for f in glob.glob(pathToRead + "**/*.png", recursive=True)]
if not os.path.exists(directory):
    os.makedirs(directory)
try :
    for f in files:
        #print(f)
        image = main(f, False)
        save_image(f, image)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
except Exception as e:
        print(e)
        print(f)    



