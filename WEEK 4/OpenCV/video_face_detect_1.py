import cv2 as cv 
import numpy as np 

def rescaleFrame(frame,scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

video = cv.VideoCapture(r'C:\Users\Shaurya\OneDrive\SOC 2026\week4\Open CV\Videos\8020479-hd_1920_1080_25fps.mp4')

haar_cascade = cv.CascadeClassifier('haarcascade.xml')

while True:
    isTrue, frame = video.read()
    frame = rescaleFrame(frame,scale=0.5)
    cv.imshow('Video', frame)

    grey_frame = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    cv.imshow('Gray_video',grey_frame)

    faces_rect = haar_cascade.detectMultiScale(grey_frame, scaleFactor=1.1 , minNeighbors=3)

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y) , (x+w,y+h) , (200,100,20) , thickness = 2 )
    
    cv.imshow('Faces detected' , frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):         #window will get closed if 'd' is pressed , by changing this 20 to 1, we can make the video play faster, by changing it to 40 we can make the video play slower
        break
    if cv.getWindowProperty('Video', cv.WND_PROP_VISIBLE) < 1:          #window will get closed if 'x' is pressed (cross button for window)
        break

