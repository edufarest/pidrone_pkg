import cv2
import numpy as np

DISTANCE_TO_LINE_CONST = 3

def findCenter(lines):
    center_x = 0
    center_y = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        center_x += x1 + x2
        center_y += y1 + y2

    center_x = center_x / (2 * len(lines))
    center_y = center_y / (2 * len(lines))

    return center_x, center_y


def isOnLine(cur_pos, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = np.asarray((x1, y1))
        p2 = np.asarray((x2, y2))
        p3 = np.asarray(cur_pos)
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        if d < DISTANCE_TO_LINE_CONST:
            return True

    return False



img = cv2.imread("images/line2.jpeg")
height, width, channels = img.shape
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([30, 50, 60])
upper_blue = np.array([255, 255, 255])
# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(img, img, mask=mask)



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(result, 140, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    print
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 10)

print isOnLine((width/2, height/2), lines)

center_x, center_y = findCenter(lines)


cv2.circle(img, (center_x, center_y), 6, [0, 0, 255], 6)
cv2.imshow("linesEdges", img)

# cv2.imshow('frame', img)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()


