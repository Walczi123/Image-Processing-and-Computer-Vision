import glob
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from descriptors import fd_hu_moments
from descriptors import fd_haralick
from descriptors import fd_histogram
from descriptors import fd_histog
from descriptors import fd_BinaryPattern
from descriptors import fd_SIFT
from summary import save_and_summary

class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]

all_descriptors = []
labels          = []
# learning from the train directories
for label in class_names:
    for file in glob.glob('train&test/'+(label)+'/train/*.jpg'):
        # read the image and resize it to the common size
        image = cv2.imread(file)
        image = cv2.resize(image, (512,512))        
        # compute descriptors
        # fv_hu_moments = fd_hu_moments(image)
        # fv_haralick   = fd_haralick(image)
        # fv_histogram  = fd_histogram(image)
        # fd_hog        = fd_histog(image)
        # fd_binpat     = fd_BinaryPattern(image)
        print(file)
        fv_SIFT       = fd_SIFT(image)  
        # save them in descriptors variable
        descriptors = np.hstack([fv_SIFT])#fv_histogram, fv_haralick, fv_hu_moments, fd_hog])
        # save labels and feature as the vectors
        labels.append(label)
        all_descriptors.append(descriptors)
    print("Computed descriptors of "+label)

# Random Forests Classifier
clf  = RandomForestClassifier(n_estimators=100, random_state=9)
# learning the model
clf.fit(all_descriptors, labels)

results = list()
print("\nPredictions\n")
for label in class_names: 
    for file in glob.glob('train&test/'+(label)+'/test/*.jpg'):
        # read the image and resize it to the common size
        image = cv2.imread(file)
        image = cv2.resize(image, (512,512))        
        # compute descriptors
        # fv_hu_moments = fd_hu_moments(image)
        # fv_haralick   = fd_haralick(image)
        # fv_histogram  = fd_histogram(image)
        # fd_hog          = fd_histog(image)
        # fd_binpat     = fd_BinaryPattern(image)
        fv_SIFT       = fd_SIFT(image)
        # save them in descriptors variable
        descriptors = np.hstack([fv_SIFT])#fv_histogram, fv_haralick, fv_hu_moments, fd_hog])
        prediction = clf.predict(descriptors.reshape(1,-1))[0]
        print(label,prediction)
        results.append((label,prediction))

save_and_summary(results)