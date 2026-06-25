import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Photos/Cat_large.webp')
img = cv.resize(img , (img.shape[1]//2, img.shape[0]//2))
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2] , dtype='uint8')
cv.imshow('blank',blank)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
# print(gray.shape)

circle = cv.circle(blank, (img.shape[1]//2-45,img.shape[0]//2-90), 100, 255, -1)
cv.imshow('circle',circle)

mask = cv.bitwise_and(img, img, mask=circle)
cv.imshow('masked' , mask)

# #grayscale histo
# gray_hist = cv.calcHist([gray],[0],mask,[256],[0,256])
# plt.figure()
# plt.title('Greyscale histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

#clour histogram
colours = ('b','g','r')
for i,col in enumerate(colours):
    hist = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()