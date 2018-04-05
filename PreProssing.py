from PIL import Image
import numpy as np
import glob
import cv2

i = 0
for filename in glob.glob('train/*.jpg'):
    img=Image.open(filename)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img.save('Small-Img/'+str(i)+'.jpg')
    i+=1


#pixels = x_data.flatten().reshape(1, 12288)

