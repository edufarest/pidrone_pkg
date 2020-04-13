import cv2
import numpy as np

DISTANCE_TO_LINE_CONST = 3
LOWER_BLUE = np.array([30, 150, 60])
UPPER_BLUE = np.array([255, 255, 190])
LOWER_RED = np.array([0, 150, 150])
UPPER_RED = np.array([10, 255, 255])

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
    return [dx * nx + x1, dy * nx + y1]

def getLines(result):
    edges = cv2.Canny(result, 100, 200)
    return cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=10)

def drawLines(lines, color):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, 5)


def getCenters(mask):
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return [cx, cy]

def getNextPosVector(red_center, blue_center):
    vector = np.subtract(red_center, blue_center)
    perp = np.cross(vector + [0], [0, 0, -1])[:2]
    return perp


class LineDetector:
    def __init__(self):
        self.cur_img = None
        self.center = None
        self.mask_blue = None
        self.mask_red = None
        self.result_blue = None
        self.result_red = None

    def updateImage(self, new_img):
        self.cur_img = new_img
        height, width, channels = img.shape
        self.center = [width / 2, height / 2]
        self.applyMask()

    def applyMask(self):
        # ========== Blue Mask ===========================
        # preparing the mask to overlay
        self.mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-blue regions
        self.result_blue = cv2.bitwise_and(self.cur_img, self.cur_img , mask=self.mask_blue)

        # ========== Red Mask ===========================
        # preparing the mask to overlay
        self.mask_red = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-red regions
        self.result_red = cv2.bitwise_and(self.cur_img, self.cur_img, mask=self.mask_red)

    def getOnlineVector(self):
        # Maintain position on line with cropped image
        red_centers = getCenters(self.mask_red)
        blue_centers = getCenters(self.mask_blue)

        # Get and return perpendicular vector
        return getNextPosVector(red_centers, blue_centers)

    def getVectorToLine(self):
        blue_lines = getLines(self.result_blue)
        red_lines = getLines(self.result_red)
        closest_blue = getClosestPointOnLine(self.center, blue_lines)
        closest_red = getClosestPointOnLine(self.center, red_lines)
        dist_blue = np.linalg.norm(np.subtract(self.center, closest_blue))
        dist_red = np.linalg.norm(np.subtract(self.center, closest_red))

        closest = closest_blue
        if dist_blue > dist_red:
            closest = closest_red

        return np.subtract(center, closest)




img = cv2.imread("images/double_line.jpeg")
height, width, channels = img.shape

# Cropping Image
img = img[height/2-100:height/2+100, width/2-100:width/2+100]
height, width, channels = img.shape
center = [width/2, height/2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



# # Get the lines
# blue_lines = getLines(result_blue)
# red_lines = getLines(result_red)
#
# # Draw lines on image
# drawLines(blue_lines, (128, 0, 0))
# drawLines(red_lines, (50, 20, 100))
#
# center_x, center_y = findCenter(red_lines)
# closest_x, closest_y = getClosestPointOnLine(center, red_lines)
#
# center_x_blue, center_y_blue = findCenter(blue_lines)
# closest_x_blue, closest_y_blue = getClosestPointOnLine(center, blue_lines)

# cv2.circle(img, (center_x, center_y), 6, [255, 0, 255], 6)
# cv2.circle(img, (closest_x, closest_y), 6, [0, 255, 255], 6)
#
# cv2.circle(img, (center_x_blue, center_y_blue), 6, [255, 0, 255], 6)
# cv2.circle(img, (closest_x_blue, closest_y_blue), 6, [0, 255, 255], 6)
#
# cv2.imshow("linesEdges", img)

# Get on the line with uncropped image


# Maintain position on line with cropped image
red_centers = getCenters(mask_red)
blue_centers = getCenters(mask_blue)

# Get perpendicular vector

nextPos = getNextPosVector(red_centers, blue_centers)

print center
print tuple(np.add(center, [int(i * 10) for i in nextPos]))

cv2.circle(img, tuple(red_centers), 6, [255, 0, 255], 6)
cv2.circle(img, tuple(blue_centers), 6, [255, 0, 255], 6)
cv2.arrowedLine(img, tuple(center), tuple(np.add(center, [int(i * 30) for i in nextPos])), [255, 0, 255], 3)

# Check if on line


cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


