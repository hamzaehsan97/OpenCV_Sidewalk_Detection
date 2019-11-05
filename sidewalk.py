import cv2
import numpy as np
import math

img = cv2.imread("/Users/hamzaehsan/Desktop/AI_Rescue_Drone/sidewalkDetection/images/test3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(gray, 55, 80)
res = cv2.bitwise_and(gray, gray, mask=mask)
edges = cv2.Canny(res, 100, 150)
lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 400, maxLineGap=40, minLineLength=700)
gradients = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 5)


cv2.imshow('mask',edges)
cv2.imshow('res',res)
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

