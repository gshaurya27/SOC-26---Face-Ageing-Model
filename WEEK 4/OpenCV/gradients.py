import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Shaurya\OneDrive\SOC 2026\week4\Open CV\Photos\Boston.webp')
img = cv.resize(img , (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Cat', img)

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('grey',gray)

#laplacian 
lap = cv.Laplacian(gray , cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)


#sobel
sobelx = cv.Sobel(gray , cv.CV_64F , 1 ,0)
sobely = cv.Sobel(gray , cv.CV_64F , 0 ,1)
combined_sobel = cv.bitwise_or(sobelx,sobely)

cv.imshow('sobelX',sobelx)
cv.imshow('sobelY',sobely)
cv.imshow('combined',combined_sobel)

canny = cv.Canny(gray,150, 175)
cv.imshow('canny',canny)



cv.waitKey(0)
cv.destroyAllWindows()