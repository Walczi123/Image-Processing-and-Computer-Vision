# import the necessary packages
import glob
import numpy as np
import cv2 
import os
from mask import mask
from labeling import label

# paths and directories
pathToRead = './multi_plant'
pathToWrite = './'
directoryMask = 'my_masks'
directoryLabel = 'my_labels'
directoryBoxes = 'my_bounding_boxes'

# save mask
def saveMasks(name):
    image = mask(name, False)
    name = os.path.basename(name)
    cv2.imwrite(pathToWrite+directoryMask+'/'+name, image)

#save label
def saveLabels(name):
    image = sum(label(name, False))
    name = os.path.basename(name)
    cv2.imwrite(pathToWrite+directoryLabel+'/'+name.replace('rgb','label'), image)
 
#save bounding box
def saveBoxes(name):
    mask = mask(name,False)
    image = cv2.imread(name,cv2.IMREAD_COLOR)
    x,y,w,h = cv2.boundingRect(mask)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    name = os.path.basename(name)
    cv2.imwrite(pathToWrite+directoryBoxes+'/'+name.replace('rgb','box'), image)

# read all files from the pathToRead
files = [f for f in glob.glob(pathToRead + "**/*.png", recursive=True)]

#if destination directory doesn't exist then create
if not os.path.exists(directoryMask):
    os.makedirs(directoryMask)
if not os.path.exists(directoryLabel):
    os.makedirs(directoryLabel) 
if not os.path.exists(directoryBoxes):
    os.makedirs(directoryBoxes)      
# counter of progress
i=0
# try to catch exceptions 
try :
    # for every file in reading directory
    for f in files:    
        # saveMasks(f)
        # saveBoxes(f)
        saveLabels(f)
        # print the progress
        print(i/9)
        i+=1
# if there is any exception print the error and the path which rise the exception        
except Exception as e:
        print(e)
        print(f)    



