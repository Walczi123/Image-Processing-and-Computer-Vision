from descriptors import fd_hu_moments
from descriptors import fd_haralick
from descriptors import fd_histogram
from descriptors import fd_histog
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# 10-fold cross validation
def cross_validation():
    print("10-fold cross validation")
    class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]
    all_descriptors = []
    labels          = []
    for label in class_names:
        for jpgfile in glob.glob("isolated/"+label+"/*.jpg"):
            image = cv2.imread(jpgfile)
            image = cv2.resize(image, (512,512))        
            # compute descriptors
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
            fd_hog        = fd_histog(image)
            # save them in descriptors variable
            descriptors = np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fd_hog])
            # save labels and feature as the vectors
            labels.append(label)
            all_descriptors.append(descriptors)
        print("Computed descriptors of "+ label)

    # # 10-fold cross validation
    kfold = KFold(n_splits=10)
    rf = RandomForestClassifier(n_estimators=100, random_state=9)
    cv_results = cross_val_score(rf, all_descriptors, labels, cv=kfold, scoring="accuracy")
    msg = "%s: %f (%f)" % ('RF', cv_results.mean(), cv_results.std())
    print(msg)

if __name__ == "__main__":
    cross_validation()   