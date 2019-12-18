import numpy as np
import cv2 
import glob
import os
from skimage.measure import compare_ssim

pathToPatterns = './multi_label/'
pathToMyPlants = './my_plants/'

def pattern(p):
    image = cv2.imread(pathToPatterns+p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
    return th1

def myPlant(p):
    image = cv2.imread(pathToMyPlants+p.replace('label','rgb'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def compare(name):
    pat = pattern(name)
    myP = myPlant(name)
    (score, diff) = compare_ssim(myP, pat, full=True)
    diff = (diff * 255).astype("uint8")
    return score

files = [f for f in glob.glob(pathToPatterns + "**/*.png", recursive=True)]
l = []
i=0
for f in files:
    name = os.path.basename(f)
    score = compare(name)
    l.append(score)
    print(round(i/9,0))
    i+=1

max_value = round(max(l) * 100, 2)
min_value = round(min(l) * 100, 2)
avg_value = round(sum(l)/len(l) * 100, 2)

print("---Structural Similarity Index----")
print("average ", avg_value)
print("minimal value ", min_value)
print("maximal value ", max_value)