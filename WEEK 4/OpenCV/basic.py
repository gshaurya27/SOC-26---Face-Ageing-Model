import cv2 as cv

img = cv.imread('Photos/Boston.webp')
# img = cv.imread(r'C:\Users\Shaurya\OneDrive\SOC 2026\week4\Open CV\Photos\Boston.webp')
img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Boston', img)

#converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur the image
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)        #the second argument is the kernel size (should be an odd number and positive), the third argument is the border type (BORDER_DEFAULT is the default border type)
cv.imshow('Blur', blur)

#edge cascade
canny = cv.Canny(blur, 125, 175)        #the second argument is the lower threshold, the third argument is the upper threshold
cv.imshow('Canny Edges', canny)

#dilating the image, dilation is a fundamental morphological operation that "grows" or thickens the bright/foreground regions/borders of an image
dilated = cv.dilate(canny, (7, 7), iterations=3)        #the second argument is the kernel size (should be an odd number and positive)
cv.imshow('Dilated', dilated)

#eroding of img
eroded = cv.erode(dilated, (7, 7), iterations=3)        #the second argument is the kernel size (should be an odd number and positive)
cv.imshow('Eroded', eroded)

#resize the image
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)        #the second argument is the new size of the image, the third argument is the interpolation method (INTER_CUBIC is a good choice for enlarging images)
cv.imshow('Resized', resized)

#cropping the image
cropped = img[50:200, 200:400]        #the first argument is the starting row, the second argument is the ending row, the third argument is the starting column, the fourth argument is the ending column
cv.imshow('Cropped', cropped)

cv.waitKey(0)
cv.destroyAllWindows()