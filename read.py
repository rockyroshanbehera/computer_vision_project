import cv2 as cv
img = cv.imread('postdetect/cat.jpeg')
cv.imshow('cat', img)
cv.waitKey(0)