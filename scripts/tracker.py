#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from apriltag_ros.msg import AprilTagDetectionArray

import apriltag
import numpy
import cv2

from cv_bridge import CvBridge

def show_image(image):
    cv2.imshow("Debug Image", image)
    cv2.waitKey(1)

def draw_detections_on_image(detections, image):
    for r in detections:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))

        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, "{}".format(r.tag_id), (cX + 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

class WorkspaceTracker(object):
    def __init__(self):
        self.bridge = CvBridge()
        options = apriltag.DetectorOptions(families="tag16h5")
        self.detector = apriltag.Detector(options)

        self.corner_ids = {
            7: 0,
            5: 1,
            6: 2,
            4: 3
        }
        x_length = 0.22
        y_length = 0.15

        self.px_per_m = 5000
        xPix = int(x_length * self.px_per_m)
        yPix = int(y_length * self.px_per_m)

        self.pts_dst = numpy.array([
            [0, 0],
            [xPix, 0],
            [0, yPix],
            [xPix, yPix]]
        )

        offset = int(0.015 * self.px_per_m)
        self.output_shape = (xPix + 2 * offset, yPix + 2 * offset)
        self.pts_dst += offset

        self.corner_locations = numpy.ones((4, 2), numpy.float32)
        self.corner_locations[:] = numpy.nan

    def new_image_message(self, img_msg):
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detections = self.detector.detect(gray)

        # show_image(draw_detections_on_image(detections, image.copy()))

        # Note that old detections are retained 
        movement = 0
        for r in detections:
            corner_id = self.corner_ids[r.tag_id]
            movement += numpy.linalg.norm(r.center - self.corner_locations[corner_id])
            self.corner_locations[corner_id] = r.center
        # print movement
        if movement > 20 and len(detections) != 4:
            return
        pts_src = numpy.array(self.corner_locations)

        h, status = cv2.findHomography(pts_src, self.pts_dst)
        trans_image = cv2.warpPerspective(image, h, self.output_shape)
    
        # Should remove the white background. Will need to do this better
        trans_image[numpy.min(trans_image, axis=2) > 100] = 0
        show_image(trans_image)

if __name__ == '__main__':
    rospy.init_node('workspace_tracker', anonymous=True)
    wt = WorkspaceTracker()
    rospy.Subscriber("/camera/color/image_rect_color", Image, wt.new_image_message)

    rospy.spin()