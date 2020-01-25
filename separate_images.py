import os
import glob
import shutil

class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]
directory = "train&test"

def separeteImages():
    # create new empty directories
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    # for every spiece
    for image in class_names:
        print("test")
        # prepare two directories 
        # one for train images
        train_dir = directory+"/"+image
        os.mkdir(train_dir)
        train_dir += "/train"
        os.mkdir(train_dir)
        # one for test images
        test_dir = directory+"/"+image+"/test"
        os.mkdir(test_dir)
        # for every image in data set
        i=0
        for jpgfile in glob.glob("isolated/"+image+"/*.jpg"):
            print("jpgfile")
            image = cv2.imread(jpgfile)
            cv2.imshow(jpgfile, image)
            cv2.waitKey(0)
            # if(i%5==0):
            #     shutil.copy(jpgfile, test_dir)
            # else:
            #     shutil.copy(jpgfile, train_dir)
            # i+=1

if __name__ == "__main__":
    separeteImages()           