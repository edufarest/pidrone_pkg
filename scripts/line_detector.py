import cv2
import numpy as np
import math

DISTANCE_TO_LINE_CONST = 3
LOWER_BLUE = np.array([80, 0, 0])
UPPER_BLUE = np.array([130, 255, 190])
LOWER_RED = np.array([150, 0, 0])
UPPER_RED = np.array([190, 255, 190])
LOWER_RED_2 = np.array([0, 0, 0])
UPPER_RED_2 = np.array([20, 255, 190])


def find_center(lines):
    if lines is None:
        return [0, 0]

    center_x = 0
    center_y = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        center_x += x1 + x2
        center_y += y1 + y2

    center_x = center_x / (2 * len(lines))
    center_y = center_y / (2 * len(lines))

    return [center_x, center_y]


def is_on_line(cur_pos, lines):
    for line in lines:
        d = get_distance_to_line(cur_pos, line)
        if d < DISTANCE_TO_LINE_CONST:
            return True

    return False


def get_distance_to_line(cur_pos, line):
    x1, y1, x2, y2 = line[0]
    p1 = np.asarray((x1, y1))
    p2 = np.asarray((x2, y2))
    p3 = np.asarray(cur_pos)
    d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return d


def get_average_angle(lines):
    if lines is None:
        return 0
    angleSum = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y1-y2, x2-x1)
        angleSum += angle

    return angleSum/len(lines)


def get_closest_line(cur_pos, lines):
    cur_min = 1000000000
    min_line = None
    for line in lines:
        d = get_distance_to_line(cur_pos, line)
        if d < cur_min:
            cur_min = d
            min_line = line

    return min_line


def get_closest_point_on_line(cur_pos, lines):
    min_line = get_closest_line(cur_pos, lines)
    x3, y3 = cur_pos
    x1, y1, x2, y2 = min_line[0]
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    nx = ((x3 - x1) * dx + (y3 - y1) * dy) / d2
    return [dx * nx + x1, dy * nx + y1]


def get_lines(result):
    edges = cv2.Canny(result, 100, 200)
    return cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=10)


def draw_lines(lines, color):
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, 2)


def get_centers(mask):
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    if moments["m00"] == 0:
        return [0, 0]

    # calculate x,y coordinate of center
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return [cx, cy]


def get_next_pos_vector(red_center, blue_center):
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

    def update_image(self, new_img):
        self.cur_img = new_img
        height, width, channels = new_img.shape
        self.center = [width / 2, height / 2]
        self.apply_mask()

    def apply_mask(self):
        hsv = cv2.cvtColor(self.cur_img, cv2.COLOR_BGR2HSV)
        # ========== Blue Mask ===========================
        # preparing the mask to overlay
        self.mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-blue regions
        self.result_blue = cv2.bitwise_and(self.cur_img, self.cur_img, mask=self.mask_blue)

        # ========== Red Mask ===========================
        # preparing the mask to overlay
        self.mask_red = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2), cv2.inRange(hsv, LOWER_RED, UPPER_RED))
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-red regions
        self.result_red = cv2.bitwise_and(self.cur_img, self.cur_img, mask=self.mask_red)

    def get_online_vector(self):
        # Maintain position on line with cropped image
        red_centers = get_centers(self.mask_red)
        blue_centers = get_centers(self.mask_blue)

        if red_centers == False or blue_centers == False:
            return [0, 0]

        # Get and return perpendicular vector
        return get_next_pos_vector(red_centers, blue_centers)

    def get_vector_to_line(self):
        blue_lines = get_lines(self.result_blue)
        red_lines = get_lines(self.result_red)
        closest_blue = get_closest_point_on_line(self.center, blue_lines)
        closest_red = get_closest_point_on_line(self.center, red_lines)
        dist_blue = np.linalg.norm(np.subtract(self.center, closest_blue))
        dist_red = np.linalg.norm(np.subtract(self.center, closest_red))

        closest = closest_blue
        if dist_blue > dist_red:
            closest = closest_red

        return np.subtract(self.center, closest)

    def get_center(self):
        return self.center

    def get_red_center(self):
        return get_centers(self.mask_red)

    def get_blue_center(self):
        return get_centers(self.mask_blue)




video = cv2.VideoCapture('images/video3.mp4')
# Check if camera opened successfully

if not video.isOpened():
  print "Error opening video stream or file"

while video.isOpened():
    ret, img = video.read()

    if ret:
        height, width, channels = img.shape
        img = img[height/2-100:height/2+100, width/2-100:width/2+100]
        line_detector = LineDetector()
        line_detector.update_image(img)
        center = line_detector.get_center()
        vector = line_detector.get_online_vector()

        print center
        print vector
        redLines = get_lines(line_detector.result_red)
        blueLines = get_lines(line_detector.result_blue)
        #draw_lines(redLines, [255, 0, 255])
        #raw_lines(blueLines, [255, 0, 255])

        angle = get_average_angle(redLines)
        angle_vector = [20 * math.cos(angle), 20 * math.sin(angle)]

        cv2.arrowedLine(img, tuple(center), tuple(np.add(center, [int(i) for i in angle_vector])), [255, 80, 255], 3)
        # cv2.circle(img, tuple(line_detector.get_red_center()), 3, [255, 0, 255], 3)
        # cv2.circle(img, tuple(line_detector.get_blue_center()), 3, [255, 0, 255], 3)
        #cv2.circle(img, tuple(find_center(redLines)), 3, [255, 0, 255], 3)
        cv2.circle(img, tuple(find_center(blueLines)), 3, [255, 0, 255], 3)
        cv2.imshow('blue', line_detector.mask_blue)
        cv2.imshow('red', line_detector.mask_red)
        cv2.imshow('frame', img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()


# img = cv2.imread("images/droneimg.png")
# height, width, channels = img.shape
# img = img[height/2-100:height/2+100, width/2-100:width/2+100]
# line_detector = LineDetector()
# line_detector.update_image(img)
# center = line_detector.get_center()
# vector = line_detector.get_online_vector()
# print center
# print vector
# cv2.arrowedLine(img, tuple(center), tuple(np.add(center, [int(i) for i in vector])), [255, 0, 255], 3)
# cv2.circle(img, tuple(line_detector.get_red_center()), 3, [255, 0, 255], 3)
# cv2.circle(img, tuple(line_detector.get_blue_center()), 3, [255, 0, 255], 3)
#
# cv2.imshow('frame', img)
# cv2.imshow('blue', line_detector.mask_blue)
# cv2.imshow('red', line_detector.mask_red)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Cropping Image
# img = img[height/2-100:height/2+100, width/2-100:width/2+100]
# height, width, channels = img.shape
# center = [width/2, height/2]
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



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


# # Maintain position on line with cropped image
# red_centers = get_centers(mask_red)
# blue_centers = get_centers(mask_blue)
#
# # Get perpendicular vector
#
# nextPos = get_next_pos_vector(red_centers, blue_centers)
#
# print center
# print tuple(np.add(center, [int(i * 10) for i in nextPos]))
#
# cv2.circle(img, tuple(red_centers), 6, [255, 0, 255], 6)
# cv2.circle(img, tuple(blue_centers), 6, [255, 0, 255], 6)
# cv2.arrowedLine(img, tuple(center), tuple(np.add(center, [int(i * 30) for i in nextPos])), [255, 0, 255], 3)

# Check if on line



