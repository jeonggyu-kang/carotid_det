# !pip install pydicom
# !pip install PyQt5
# !pip install python-gdcm
# !pip install opencv-python

import os, cv2, sys
from PyQt5.QtWidgets import *
import datetime
import pydicom as dc
import numpy as np

file_name = './59.dcm' # file혹은 dir 선택

dcm = dc.filereader.dcmread(file_name)
# print (dcm)
image = dcm.pixel_array
print (image.shape)

img_Gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )

img_crop = img_Gray[10:350, 110:525].copy()

print (type(img_Gray))

'''
s = np.einsum('ijk->j', image)
t = np.einsum('ijk->i', image)
print (s, len(s))
print (t, len(t))
'''

sobelx = cv2.Sobel(image, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(image, -1, 0, 1, ksize=3)


save_file = file_name.replace('.dcm','.bmp')

cv2.imwrite("58.bmp",img_Gray)
cv2.imwrite(save_file,img_crop)
cv2.imwrite("x.bmp", sobelx)
cv2.imwrite("y.bmp", sobely)