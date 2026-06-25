import cv2 as cv 
import numpy as np

img = cv.imread('Photos/Cat_large.webp')
img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Cat', img)


blank = np.zeros(img.shape, dtype='uint8')      
cv.imshow('Blank', blank) 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)        
cv.imshow('Canny', canny)

# ret, thresh = cv.threshold(gray,              # source image
#                             125,               # threshold value, above this value will be set to max value, below this value will be set to 0
#                             255,               # maximum value
#                             cv.THRESH_BINARY   # threshold type
#                             )
# cv.imshow('Thresholded', thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)        #the second argument is the contour retrieval mode (RETR_LIST retrieves all contours without establishing any hierarchical relationships), the third argument is the contour approximation method (CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)        #the first argument is the image on which we want to draw the contours, the second argument is the contours we want to draw, the third argument is the index of the contour we want to draw (-1 means draw all contours), the fourth argument is the colour of the contour in BGR format, and the fifth argument is the thickness of the contour
cv.imshow('Contours Drawn', blank)


cv.waitKey(0)
cv.destroyAllWindows()
