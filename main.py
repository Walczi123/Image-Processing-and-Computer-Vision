import glob
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from descriptors import fd_hu_moments
from descriptors import fd_haralick
from descriptors import fd_histogram
from summary import save_and_summary

# num_trees = 100
# seed      = 9

# # 10-fold cross validation
# kfold = KFold(n_splits=10, random_state=seed)
# rf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
# cv_results = cross_val_score(rf, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="accuracy")
# msg = "%s: %f (%f)" % ('RF', cv_results.mean(), cv_results.std())
# print(msg)

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
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        # save them in descriptors variable
        descriptors = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        # save labels and feature as the vectors
        labels.append(label)
        all_descriptors.append(descriptors)
    print("[STATUS] processed folder: {}".format(label))

print("[STATUS] completed Global Feature Extraction")

# Random Forests Classifier
clf  = RandomForestClassifier(n_estimators=100, random_state=9)
# learning the model
clf.fit(all_descriptors, labels)

results = list()

for label in class_names: 
    for file in glob.glob('train&test/'+(label)+'/test/*.jpg'):
        # read the image and resize it to the common size
        image = cv2.imread(file)
        image = cv2.resize(image, (512,512))        
        # compute descriptors
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        # save them in descriptors variable
        descriptors = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        prediction = clf.predict(descriptors.reshape(1,-1))[0]
        print(label,prediction)
        results.append((label,prediction))

save_and_summary(results)