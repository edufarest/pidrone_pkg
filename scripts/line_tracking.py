from __future__ import division
import cv2
from pidrone_pkg.msg import State
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Pose
from analyze_flow import AnalyzeFlow
from cv_bridge import CvBridge
import rospy
import picamera
import picamera.array
import numpy as np

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_CENTER = np.float32([(CAMERA_WIDTH - 1) / 2., (CAMERA_HEIGHT - 1) / 2.])
DISTANCE_TO_LINE_CONST = 3


class ObjectTracker(picamera.array.PiMotionAnalysis):
    def __init__(self, camera):
        picamera.array.PiMotionAnalysis.__init__(self, camera)

        self.obj_pose_pub = rospy.Publisher('/pidrone/desired/pose', Pose, queue_size=1)


        self.curr_obj_coordinates = None
        self.error = None
        self.track_object = False

        self.prev_img = None

        self.alpha = 0.70
        self.z = 0.16

    def write(self, data):
        curr_img = np.reshape(np.fromstring(data, dtype=np.uint8), (CAMERA_HEIGHT, CAMERA_WIDTH, 3))

        # start object tracking
        if self.track_object:


        self.prev_img = curr_img

    def reset_callback(self, data):
        if not self.track_object:
            print "Start tracking object"
        else:
            print "Stop tracking object"

        self.track_object = not self.track_object
        self.curr_obj_coordinates = None

    def state_callback(self, data):
        self.z = data.pose_with_covariance.pose.position.z

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
    return perp / np.linalg.norm(perp)


def find_densest_point(x, y, r=50, max_density_point=None, max_density=None, iteration=0):

    while iteration < 2:

        points_x = x
        points_y = y
        iteration += 1

        num_points = len(points_x)

        z = np.array([complex(points_x[i], points_y[i]) for i in range(num_points)])
        distances = abs(z[..., np.newaxis]-z)

        densities = np.exp((-distances ** 2) / (r**2)).sum(axis=1)
        max_density_index = densities.argmax()
        max_density_point = np.array([points_x[max_density_index], points_y[max_density_index]])
        max_density = densities.max()

        local_r = 50.0

        mask = distances[max_density_index] <= local_r
        l_points_x = points_x[mask]
        l_points_y = points_y[mask]

        return find_densest_point(l_points_x, l_points_y, 50, max_density_point, max_density, iteration)

    return max_density_point, max_density


def main():
    rospy.init_node("object_tracking")

    image_pub = rospy.Publisher("/pidrone/picamera/image_raw", Image, queue_size=1, tcp_nodelay=False)

    try:

        bridge = CvBridge()

        with picamera.PiCamera(framerate=90) as camera:
            camera.resolution = (320, 240)
            with ObjectTracker(camera) as tracker:

                rospy.Subscriber("/pidrone/reset_transform", Empty, tracker.reset_callback)
                rospy.Subscriber("/pidrone/state", State, tracker.state_callback)

                with AnalyzeFlow(camera) as flow_analyzer:
                    # run the setup functions for each of the image callback classes
                    flow_analyzer.setup(camera.resolution)

                    # start the recordings for the image and the motion vectors
                    camera.start_recording("/dev/null", format='h264', splitter_port=1, motion_output=flow_analyzer)
                    camera.start_recording(tracker, format='bgr', splitter_port=2)

                    while not rospy.is_shutdown():
                        camera.wait_recording(1 / 100.0)

                        if tracker.prev_img is not None:
                            image_message = bridge.cv2_to_imgmsg(tracker.prev_img, encoding="bgr8")
                            image_pub.publish(image_message)

                camera.stop_recording(splitter_port=1)
            camera.stop_recording(splitter_port=2)
        print "Shutdown Received"
    except Exception:
        print "Camera Error!!"
        raise


if __name__ == "__main__":
    main()
