import cv2 
import mahotas
import numpy as np
from skimage import feature

# Hu Moments
def fd_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature
    
# Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# Color Histogram
def fd_histogram(image):
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Histogram of Oriented Gradients
def fd_histog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return h

def fd_orb(image):
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    image=cv2.drawKeypoints(gray,kp,image)
    cv2.show('sift_keypoints.jpg',image)

def fd_SIFT(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    freakExtractor = cv2.xfeatures2d.FREAK_create()
    keypoints,descriptors= freakExtractor.compute(gray,None)
    return keypoints 

def fd_BinaryPattern(image):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, 24, 8, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, 27),
        range=(0, 26))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # return the histogram of Local Binary Patterns
    return hist

# for tests
if __name__ == "__main__":
    image = cv2.imread(".\isolated\negundo\l20.jpg")
    cv2.imshow("test", cv2.imread("C:\Patryk\GitHub Repository\Image-Processing-and-Computer-Vision-Project-II\isolated123\negundo\l20.jpg"))
    cv2.waitKey(0)
    # print(fd_SIFT(image))  