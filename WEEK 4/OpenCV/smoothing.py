import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('Photos/Boston.webp')
img = cv.resize(img , (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Boston', img)

#averaging
average = cv.blur(img,(7,7))
cv.imshow('Average', average)

#Gaussian(done with weights)
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian', gauss)

#Median
median = cv.medianBlur(img, 7)
cv.imshow('Median', median)

#Bilateral bluring
bilateral = cv.bilateralFilter(img, 20, 15, 150)
cv.imshow('bilateral' ,bilateral)



cv.waitKey(0)
cv.destroyAllWindows()
