import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')      #uint8 is dtype of image, 8 bit unsigned integer, 0-255, 3 is number of colour channels
cv.imshow('Blank', blank)

#1 Paint the image a certain colour
# blank[200:300, 100:400] = 0, 225, 0        # BGR format, so this is green , area of height 200-300 and width 100-400 will be painted green
# cv.imshow('Green', blank)

#draw a rectangle
cv.rectangle(blank, (0,0),(250, 250), (0, 0, 255), thickness=2)        #thickness=(-1) or (cv.FILLED) will fill the rectangle with colour
#the first argument is the image on which we want to draw the rectangle, second argument is the starting point of the rectangle, third argument is the ending point of the rectangle, fourth argument is the colour of the rectangle in BGR format, fifth argument is the thickness of the rectangle
cv.imshow('Rectangle', blank)

#Draw a circle
cv.circle(blank , (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 255, 0), thickness=-1)        
#the first argument is the image on which we want to draw the circle, second argument is the center of the circle, third argument is the radius of the circle, fourth argument is the colour of the circle in BGR format, fifth argument is the thickness of the circle
cv.imshow('Circle', blank)

#draw a line
cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)        
#the first argument is the image on which we want to draw the line, second argument is the starting point of the line, third argument is the ending point of the line, fourth argument is the colour of the line in BGR format, fifth argument is the thickness of the line
cv.imshow('Line', blank)

#put text on image
cv.putText(blank, 'Hello World', (225, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (225, 0, 0), thickness=2)
#the first argument is the image on which we want to put the text, second argument is the text we want to put, third argument is the starting point of the text, fourth argument is the font of the text, fifth argument is the font scale, sixth argument is the colour of the text in BGR format, seventh argument is the thickness of the text 
cv.imshow('Text', blank)

cv.waitKey(0)
cv.destroyAllWindows()
