# import the necessary packages
import numpy as np
import cv2 
import glob
import os
from skimage.measure import compare_ssim
from sklearn.metrics import jaccard_similarity_score
from mask import mask

# paths and directories
pathToPatterns = './multi_label/'
pathToMyLabels = './my_labels/'
pathToResults = './ComparisionLabels/'

colors = [(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(128,128,128),(128,0,0)]

# read the pattern
def pattern(path):
    image = cv2.imread(pathToPatterns+path)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    d = dict()
    for c in colors:
        arr = np.array(c)
        res = cv2.inRange(image, arr, arr)
        if np.sum(res) > 0:
            segmented = cv2.bitwise_and(image , image , mask=res)
            d[c] = segmented
    return d

# read my mask
def my_label(path, keys):
    image = cv2.imread(pathToMyLabels+path)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR) 
    d = dict()
    for c in keys:
        arr = np.array(c)
        res = cv2.inRange(image, arr, arr)
        segmented = cv2.bitwise_and(image , image , mask=res)
        d[c] = segmented
    return d

# IoU comparison
def IoU_compare(my_mask,pat):
    img1 = np.asarray(pat).astype(np.bool)
    img2 = np.asarray(my_mask).astype(np.bool)
    num = np.sum(np.bitwise_and(img1,img2))
    den = np.sum(np.bitwise_or(img1,img2))
    return num/den

# dice coefficient coparison
def dice_coeff_compare(my_mask, pat):
    img1 = np.asarray(pat).astype(np.bool)
    img2 = np.asarray(my_mask).astype(np.bool)
    img_intersection = np.logical_and(img1, img2)
    image_sum = img1.sum() + img2.sum()
    if image_sum == 0:
        return 0
    return 2. * img_intersection.sum() / image_sum 

# read all files
files = [f for f in glob.glob(pathToPatterns + "**/*.png", recursive=True)]
# prepare the empty dictionaries
dice = dict()
ssim = dict()
jacc = dict()
iou = dict()

if not os.path.exists(pathToResults):
    os.makedirs(pathToResults)

completeName = os.path.join(pathToResults, "Dice.txt")         
file_dice = open(completeName , 'w') 
completeName = os.path.join(pathToResults, "IoU.txt")         
file_iou = open(completeName , 'w') 

# set counter of progress
i=0
# try to catch exceptions 
try :
    # for every file in reading directory
    for f in files:
        name = os.path.basename(f)
        pat = pattern(name) 
        my_l = my_label(name, pat.keys())
        s=0
        j=0
        d=0
        u=0
        for key in pat.keys():
            dice[str(name)+str(key)] = dice_coeff_compare(my_l[key], pat[key])
            iou[str(name)+str(key)] = IoU_compare(my_l[key], pat[key])
            d += dice[str(name)+str(key)]
            u += iou[str(name)+str(key)]
            file_dice.write(str(name)+" "+str(key)+" "+str(dice[str(name)+str(key)]*100)+"\n") 
            file_iou.write(str(name)+" "+str(key)+" "+str(iou[str(name)+str(key)]*100)+"\n") 
        dice[str(name)] = d/len(pat)
        iou[str(name)] = u/len(pat)
        file_dice.write(str(name)+" "+str(dice[str(name)]*100)+"\n") 
        file_iou.write(str(name)+" "+str(iou[str(name)]*100)+"\n") 
        # print the progress
        print(round(i/9,0))
        i+=1 
    
    # compute the results for IoU metric
    key_max = max(iou.keys(), key=(lambda k: iou[k]))
    key_min = min(iou.keys(), key=(lambda k: iou[k]))
    avg_value = np.array([iou[key] for key in iou]).mean()

    # print the result for IoU metric
    print("--- IoU metric ----")
    print("average ", round(avg_value*100,2))
    print("minimal value ", round(iou[key_min]*100,2)," ",key_min)
    print("maximal value ", round(iou[key_max]*100,2)," ",key_max)

    file_iou.write("--- IoU metric ----\n")
    file_iou.write("average "+str(round(avg_value*100,2))+"\n")
    file_iou.write("minimal value "+str(round(iou[key_min]*100,2))+" "+str(key_min)+"\n")
    file_iou.write("maximal value "+str(round(iou[key_max]*100,2))+" "+str(key_max)+"\n")
  
    # compute the results for Dice coefficient
    key_max = max(dice.keys(), key=(lambda k: dice[k]))
    key_min = min(dice.keys(), key=(lambda k: dice[k]))
    avg_value = np.array([dice[key] for key in dice]).mean()

    # print the result for Dice coefficient
    print("--- Dice coefficient ----")
    print("average ", round(avg_value*100,2))
    print("minimal value ", round(dice[key_min]*100,2)," ",key_min)
    print("maximal value ", round(dice[key_max]*100,2)," ",key_max)

    file_dice.write("--- Dice coefficient ----\n")
    file_dice.write("average "+str(round(avg_value*100,2))+"\n")
    file_dice.write("minimal value "+str(round(dice[key_min]*100,2))+" "+str(key_min)+"\n")
    file_dice.write("maximal value "+str(round(dice[key_max]*100,2))+" "+str(key_max)+"\n")

# if there is any exception print the error and the path which rise the exception    
except Exception as e:
        print(e)
        print(f) 