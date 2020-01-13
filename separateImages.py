import os
import glob
import shutil

class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]
directory = "train&test"

def separeteImages():
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    for image in class_names:
        train_dir = directory+"/"+image
        os.mkdir(train_dir)
        train_dir += "/train"
        os.mkdir(train_dir)
        test_dir = directory+"/"+image+"/test"
        os.mkdir(test_dir)
        i=0
        for jpgfile in glob.glob("isolated/"+image+"/*.jpg"):
            if(i%5==0):
                shutil.copy(jpgfile, test_dir)
                # os.rename(file,file.split("\\")[0]+"/test/"+file.split("\\")[1])
            else:
                shutil.copy(jpgfile, train_dir)
                # os.rename(file,file.split("\\")[0]+"/train/"+file.split("\\")[1])
            i+=1

if __name__ == "__main__":
    separeteImages()           