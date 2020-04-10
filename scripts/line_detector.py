import cv2
import numpy as np
import numpy.linalg

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
    x, y = cur_pos

    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = np.asarray((x1, y1))
        p2 = np.asarray((x2, y2))
        p3 = np.asarray(cur_pos)
        d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)


def distanceBetweenLines(pos_1, pos_2):
    x, y = pos_1
    x1, y1 = pos_2
    return (x1 - x) ^ 2 + (y1 - y)^2

img = cv2.imread("images/line.jpeg")
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
    print line
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 10)


center_x, center_y = findCenter(lines)


cv2.circle(img, (center_x, center_y), 6, [0, 0, 255], 6)
cv2.imshow("linesEdges", img)

# cv2.imshow('frame', img)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()


