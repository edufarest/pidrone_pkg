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
        d = getDistanceToLine(cur_pos, line)
        if d < DISTANCE_TO_LINE_CONST:
            return True

    return False

def getDistanceToLine(cur_pos, line):
    x1, y1, x2, y2 = line[0]
    p1 = np.asarray((x1, y1))
    p2 = np.asarray((x2, y2))
    p3 = np.asarray(cur_pos)
    d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return d

def getClosestLine(cur_pos, lines):
    cur_min = 1000000000
    min_line = None
    for line in lines:
        d = getDistanceToLine(cur_pos, line)
        if d < cur_min:
            cur_min = d
            min_line = line

    return min_line


def getClosestPointOnLine(cur_pos, lines):
    min_line = getClosestLine(cur_pos, lines)
    x3, y3 = cur_pos
    x1, y1, x2, y2 = min_line[0]
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    nx = ((x3 - x1) * dx + (y3 - y1) * dy) / d2
    return (dx * nx + x1, dy * nx + y1)


img = cv2.imread("images/line.jpeg")
height, width, channels = img.shape
center = (width/2, height/2)
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

center_x, center_y = findCenter(lines)
closest_x, closest_y = getClosestPointOnLine(center, lines)

cv2.circle(img, (center_x, center_y), 6, [0, 0, 255], 6)
cv2.circle(img, (closest_x, closest_y), 6, [0, 255, 255], 6)

cv2.imshow("linesEdges", img)

# cv2.imshow('frame', img)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()


