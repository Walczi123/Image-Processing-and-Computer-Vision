import glob
import numpy as np
import cv2 
from main_code import main

path = './multi_plant'

  

files = [f for f in glob.glob(path + "**/*.png", recursive=True)]
for f in files:
    #print(f)
    main(f, True)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()