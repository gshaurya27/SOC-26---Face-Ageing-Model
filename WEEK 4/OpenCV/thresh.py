import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Photos/Cat_large.webp')
img = cv.resize(img , (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Cat', img)

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('grey',gray)

#simple thresholding 
threshold , thresh = cv.threshold(gray, 150 , 255, cv.THRESH_BINARY)
cv.imshow('Simple thresholded',thresh)

threshold , thresh_inv = cv.threshold(gray, 150 , 255, cv.THRESH_BINARY_INV)            #black and white inveresed
cv.imshow('Simple thresholded',thresh_inv)

#adaptic threshholding
adaptic_thresh = cv.adaptiveThreshold(gray , 255, cv.ADAPTIVE_THRESH_MEAN_C , cv.THRESH_BINARY , 11 , 5 )
cv.imshow('adapticthresh',adaptic_thresh)



cv.waitKey(0)
cv.destroyAllWindows()