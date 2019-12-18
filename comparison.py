import numpy as np
import cv2 
import glob

pathToPatterns = './multi_label'
pathToMyPlants = './my_plants'

def pattern(p):
    image = cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    ret,th1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
    image_true = th1
    image_pred = l_contours
    image_true=np.array(image_true).ravel()
    image_pred=np.array(image_pred).ravel()
    iou = sm.jaccard_similarity_score(image_true, image_pred)
    print(iou)

    cv2.waitKey()

files = [f for f in glob.glob(pathToPatterns + "**/*.png", recursive=True)]
for f in files:
    pattern(f)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break   