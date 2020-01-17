import cv2.cv2 as cv
import numpy as np

print(cv.__version__)
img = cv.imread('Cat03.jpg')
px = img[100,100]
print(px)

# accessing only blue pixel
blue = img[100,100,0]
print(blue)

print(img.size)
print(img.dtype)