# import the necessary packages
import numpy as np
import cv2 
import glob
import os
from skimage.measure import compare_ssim
from sklearn.metrics import jaccard_similarity_score

# paths and directories
pathToPatterns = './multi_label/'
pathToMyMasks = './my_masks/'
pathToResults = './ComparisionMasks/'

# read the pattern
def pattern(path):
    image = cv2.imread(pathToPatterns+path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
    # return binary mask
    return th1

# read my mask
def my_mask(path):
    image = cv2.imread(pathToMyMasks+path.replace('label','rgb'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# ssim coparison
def ssim_compare(my_mask, pat):
    (score, diff) = compare_ssim(my_mask, pat, full=True)
    diff = (diff * 255).astype("uint8")
    return score

# jaccard comaprison by function
def jaccard_compare(my_mask, pat):
    score = jaccard_similarity_score(my_mask.flatten(), pat.flatten())
    return score   

# IoU comparison
def IoU_compare(my_mask,pat):
    img1 = np.asarray(pat).astype(np.bool)
    img2 = np.asarray(my_mask).astype(np.bool)
    num = np.sum(np.logical_and(img1,img2))
    den = np.sum(np.logical_or(img1,img2))
    return num/den

# dice coefficient coparison
def dice_coeff_compare(my_mask, pat):
    img1 = np.asarray(pat).astype(np.bool)
    img2 = np.asarray(my_mask).astype(np.bool)
    img_intersection = np.bitwise_and(img1, img2)
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
completeName = os.path.join(pathToResults, "Ssim.txt")         
file_ssim = open(completeName , 'w') 
completeName = os.path.join(pathToResults, "Jacc.txt")         
file_jacc = open(completeName , 'w') 
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
        my_m = my_mask(name)
        name = name.replace('label','rgb')
        ssim[name] = ssim_compare(my_m, pat)
        jacc[name] = jaccard_compare(my_m, pat)
        dice[name] = dice_coeff_compare(my_m, pat)
        iou[name] = IoU_compare(my_m, pat)
        # save to files
        file_dice.write(str(name)+" "+str(dice[name]*100)+"\n") 
        file_jacc.write(str(name)+" "+str(jacc[name]*100)+"\n") 
        file_ssim.write(str(name)+" "+str(ssim[name]*100)+"\n") 
        file_iou.write(str(name)+" "+str(iou[name]*100)+"\n") 
        # print the progress
        print(round(i/9,0))
        i+=1 
            
    # compute the results for Jaccard Index
    key_max = max(jacc.keys(), key=(lambda k: jacc[k]))
    key_min = min(jacc.keys(), key=(lambda k: jacc[k]))
    avg_value = np.array([jacc[key] for key in jacc]).mean()

    # print the result for Jaccard Index
    print("---Jaccard Index----")
    print("average ", round(avg_value*100,2))
    print("minimal value ", round(jacc[key_min]*100,2)," ",key_min)
    print("maximal value ", round(jacc[key_max]*100,2)," ",key_max)

    file_jacc.write("---Jaccard Index----\n")
    file_jacc.write("average "+str(round(avg_value*100,2))+"\n")
    file_jacc.write("minimal value "+str(round(jacc[key_min]*100,2))+" "+str(key_min)+"\n")
    file_jacc.write("maximal value "+str(round(jacc[key_max]*100,2))+" "+str(key_max)+"\n")

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

    # compute the results for Structural Similarity Index
    key_max = max(ssim.keys(), key=(lambda k: ssim[k]))
    key_min = min(ssim.keys(), key=(lambda k: ssim[k]))
    avg_value = np.array([ssim[key] for key in ssim]).mean()

    # print the result for Structural Similarity Index
    print("---Structural Similarity Index----")
    print("average ", round(avg_value*100,2))
    print("minimal value ", round(ssim[key_min]*100,2)," ",key_min)
    print("maximal value ", round(ssim[key_max]*100,2)," ",key_max)

    file_ssim.write("---Structural Similarity Index----\n")
    file_ssim.write("average "+str(round(avg_value*100,2))+"\n")
    file_ssim.write("minimal value "+str(round(ssim[key_min]*100,2))+" "+str(key_min)+"\n")
    file_ssim.write("maximal value "+str(round(ssim[key_max]*100,2))+" "+str(key_max)+"\n")

    file_dice.close()    
    file_jacc.close() 
    file_ssim.close() 
    file_iou.close() 

# if there is any exception print the error and the path which rise the exception    
except Exception as e:
        print(e)
        print(f) 