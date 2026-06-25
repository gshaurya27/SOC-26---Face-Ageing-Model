import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('Photos/Cat_large.webp')
img = cv.resize(img , (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2] , dtype='uint8')
cv.imshow('blank',blank)

mask = cv.circle(blank, (img.shape[1]//2-45,img.shape[0]//2-80), 100, 255, -1)
cv.imshow('mask',mask)

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (img.shape[1]//2-45,img.shape[0]//2-80), 100, 255, -1)

b_and = cv.bitwise_and(rectangle, circle)
cv.imshow('bitwise_and', b_and)

masked = cv.bitwise_and(img,img,mask=b_and)
cv.imshow('masked',masked)



cv.waitKey(0)
cv.destroyAllWindows()