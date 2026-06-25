import cv2 as cv 
import numpy as np

blank = np.zeros((400,400) , dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Rec' , rectangle)
cv.imshow('Circle', circle)

#bitwise and 
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('bitwise_and', bitwise_and)

#bitwise OR
b_or = cv.bitwise_or(rectangle,circle)
cv.imshow('OR', b_or)

#bitwise XOR
b_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow('XOR', b_xor)

#bitwise not
b_not = cv.bitwise_not(rectangle)
cv.imshow('NOT',b_not)



cv.waitKey(0)
cv.destroyAllWindows()
