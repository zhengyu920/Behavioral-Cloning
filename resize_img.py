# this program resize a RGB image to 160*320 in shape

import cv2
import numpy as np
img = cv2.imread('images.jpeg')
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#OR
height, width = img.shape[:2]
res = cv2.resize(img,(320, 160), interpolation = cv2.INTER_CUBIC)
cv2.imwrite("road.jpg", res)