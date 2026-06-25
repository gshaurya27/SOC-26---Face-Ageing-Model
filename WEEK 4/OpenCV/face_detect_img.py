import cv2 as cv 
import numpy as np 

img = cv.imread('Photos/Group 2.jpeg')
img = cv.resize(img , (img.shape[1]*2,img.shape[0]*2))
cv.imshow('lady',img)

gray = gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('grey',gray)

haar_cascade = cv.CascadeClassifier('haarcascade.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=2.0 , minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y) , (x+w,y+h) , (255,255,0) , thickness = 2 )

cv.imshow('detected faces' , img)

cv.waitKey(0)
cv.destroyAllWindows()