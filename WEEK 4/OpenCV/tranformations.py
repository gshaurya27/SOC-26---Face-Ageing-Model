import cv2 as cv
import numpy as np

img = cv.imread('Photos/Boston.webp')

cv.imshow('Boston', img)

#Translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])        #the first row is for x direction and the second row is for y direction
    dimensions = (img.shape[1], img.shape[0])        #the first argument is the width and the second argument is the height
    return cv.warpAffine(img, transMat, dimensions)       
        #warpAffine is a function that applies an affine transformation to an image, the first argument is the image, the second argument is the transformation matrix, and the third argument is the dimensions of the output image
        # -x --> Moves left, +x --> Moves right, -y --> Moves up, +y --> Moves down

img_translated = translate(img, 100, 100)
cv.imshow('Translated', img_translated) 

#Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]        #the first argument is the height and the second argument is the width
    if rotPoint is None:
        rotPoint = (width//2, height//2)        #the first argument is the x coordinate and the second argument is the y coordinate
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)        #the first argument is the center of rotation, the second argument is the angle of rotation(acw if positive), and the third argument is the scale factor
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

#resizing the image
resized = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv.INTER_CUBIC)        #the second argument is the new size of the image, the third argument is the interpolation method (INTER_CUBIC is a good choice for enlarging images))
cv.imshow('Resized', resized)

#flipping the image
flipped = cv.flip(img, 0)        #the second argument is the flip code (0 for vertical flip, 1 for horizontal flip, -1 for both)
cv.imshow('Flipped', flipped)


cv.waitKey(0)
cv.destroyAllWindows()


