import cv2
import numpy as np
from skimage.morphology import skeletonize

# Read image
img = cv2.imread('c.jpg')
hh, ww = img.shape[:2]

# convert to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

# invert so shapes are white on black background
thresh = 255 - thresh

# get contours and save area
cntrs_info = []
contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
index=0
for cntr in contours:
    area = cv2.contourArea(cntr)
    cntrs_info.append((index,area))
    index = index + 1

# sort contours by area
def takeSecond(elem):
    return elem[1]
cntrs_info.sort(key=takeSecond, reverse=True)

# get third largest contour
arms = np.zeros_like(thresh)
index_third = cntrs_info[2][0]
cv2.drawContours(arms,[contours[index_third]],0,(1),-1)

#arms=cv2.ximgproc.thinning(arms)
arms_thin = skeletonize(arms)
arms_thin = (255*arms_thin).clip(0,255).astype(np.uint8)

# get hough lines and draw on copy of input
result = img.copy()
lineThresh = 15
minLineLength = 20
maxLineGap = 100
max
lines = cv2.HoughLinesP(arms_thin, 1, np.pi/180, lineThresh, None, minLineLength, maxLineGap)

for [line] in lines:
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 2)   

# save results
cv2.imwrite('clock_thresh.jpg', thresh)
cv2.imwrite('clock_arms.jpg', (255*arms).clip(0,255).astype(np.uint8))
cv2.imwrite('clock_arms_thin.jpg', arms_thin)
cv2.imwrite('clock_lines.jpg', result)

cv2.imshow('thresh', thresh)
cv2.imshow('arms', (255*arms).clip(0,255).astype(np.uint8))
cv2.imshow('arms_thin', arms_thin)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()