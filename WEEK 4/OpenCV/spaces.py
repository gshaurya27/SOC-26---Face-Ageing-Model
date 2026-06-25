import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('Photos/Cat_large.webp')
img = cv.resize(img , (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Cat', img)

#BGR to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)   

#BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

plt.imshow(rgb)         #this imge will be shown in the correct color format because matplotlib uses RGB format and OpenCV uses BGR format, so the red and blue channels are swapped
plt.show()

#HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV to BGR', hsv_bgr)

#L*a*b to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB to BGR', lab_bgr)

###there is no direct way to covert from greyscale to HSV or L*a*b, so we have to first convert it to BGR and then to HSV or L*a*b

# plt.imshow(img)       #here there is an inversion of image because matplotlib uses RGB format and OpenCV uses BGR format, so the red and blue channels are swapped
# plt.show()



cv.waitKey(0)
cv.destroyAllWindows()
