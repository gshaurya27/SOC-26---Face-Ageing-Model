import os 
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade.xml')

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recogniser = cv.face.LBPHFaceRecognizer()
face_recogniser.read('face_trained.yml')

img = cv.imread('any photo path from these people')

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('person',img)

#detect face
faces_rect = haar_cascade.detectMultiScale(gray, 1.1 , 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    labels, confidence = face_recogniser.predict(faces_roi)
    print(f'Label = {people[lable]} with a confidence of {confidence}')      

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0),thinkness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected', img)
cv.imshow(0) 
