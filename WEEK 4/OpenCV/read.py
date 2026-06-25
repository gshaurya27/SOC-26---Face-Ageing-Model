import cv2 as cv

def rescaleFrame(frame,scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    # Live Video only
    video.set(3,width)            # Set width to 3, width = 1280 pixels
    video.set(4,height)           # Set height to 4, height = 720 pixels

# cat_img = cv.imread(r'C:\Users\Shaurya\OneDrive\SOC 2026\week4\Open CV\Photos\Cat.jpg')
# cv.imshow('Cat', cat_img)
# cv.waitKey(0)

# cat_large = cv.imread(r'C:\Users\Shaurya\OneDrive\SOC 2026\week4\Open CV\Photos\Cat_large.webp')
# cat_large = rescaleFrame(cat_large,scale=0.5)
# cv.imshow('Cat Large', cat_large)
# cv.waitKey(0)


#Reading VIdeos

video = cv.VideoCapture('C:\\Users\\Shaurya\\OneDrive\\SOC 2026\\week4\\Open CV\\Videos\\7802455-uhd_4096_2160_25fps.mp4')


while True:
    isTrue, frame = video.read()
    frame = rescaleFrame(frame,scale=0.2)
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):         #window will get closed if 'd' is pressed , by changing this 20 to 1, we can make the video play faster, by changing it to 40 we can make the video play slower
        break
    if cv.getWindowProperty('Video', cv.WND_PROP_VISIBLE) < 1:          #window will get closed if 'x' is pressed (cross button for window)
        break

video.release()
cv.destroyAllWindows()
