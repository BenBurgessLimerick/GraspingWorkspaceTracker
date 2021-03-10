#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo

DEBUG = False
if DEBUG:
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pcl2

import std_msgs.msg

import apriltag
import numpy
import cv2
import tf2_ros
import geometry_msgs.msg
import tf_conversions

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
        # cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        # cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        # cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        # cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))

        cv2.circle(image, (cX, cY), 1, (0, 0, 255), -1)
        # draw the tag family on the image
        # cv2.putText(image, "{}".format(r.tag_id), (cX + 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

class WorkspaceTracker(object):
    def __init__(self, tag_size=0.03, x_length=0.225, y_length=0.14):
        self.bridge = CvBridge()
        options = apriltag.DetectorOptions(families="tag16h5")
        self.detector = apriltag.Detector(options)

        # TODO: This needs to map from tag ids to corner locations. Should probably learn it on first image rather than specify
        self.corner_ids = {
            7: 0,
            5: 1,
            6: 2,
            4: 3
        }

        self.tag_size = tag_size
        self.x_length = x_length
        self.y_length = y_length

        self.px_per_m = 5000
        xPix = int(self.x_length * self.px_per_m)
        yPix = int(self.y_length * self.px_per_m)

        self.corner_coords = numpy.array([
            [-self.x_length/2, -self.y_length/2, 0],
            [self.x_length/2, -self.y_length/2, 0], 
            [-self.x_length/2, self.y_length/2, 0], 
            [self.x_length/2, self.y_length/2, 0]
        ])

        self.pts_dst = numpy.array([
            [0, 0],
            [xPix, 0],
            [0, yPix],
            [xPix, yPix]]
        )

        offset = int(self.tag_size / 2 * self.px_per_m)
        self.output_shape = (xPix + 2 * offset, yPix + 2 * offset)
        self.pts_dst += offset

        self.corner_locations = numpy.empty((4, 2), numpy.float32)
        self.corner_locations[:] = numpy.nan

        self.corner_positions = numpy.empty((4, 3), numpy.float32)
        self.corner_positions[:] = numpy.nan

        if DEBUG:
            self.pc_pub = rospy.Publisher("/test_cloud", PointCloud2, queue_size=10)
            self.pc_pub2 = rospy.Publisher("/test_cloud2", PointCloud2, queue_size=10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.transformed_workspace_pub = rospy.Publisher('transformed_workspace', Image, queue_size=10)

        self.camera_params = None

    def new_image_message(self, img_msg):
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detections = [d for d in self.detector.detect(gray) if d.tag_id in self.corner_ids]

        # show_image(draw_detections_on_image(detections, image.copy()))

        # Note that old detections are retained 
        movement = 0

        for r in detections:
            if r.tag_id not in self.corner_ids:
                del r
                continue
            corner_id = self.corner_ids[r.tag_id]
            movement += numpy.linalg.norm(r.center - self.corner_locations[corner_id])
            self.corner_locations[corner_id] = r.center

            if self.camera_params:
                pose, e0, e1 = self.detector.detection_pose(r, self.camera_params, self.tag_size)
                pos = pose[0:3, 3]
                self.corner_positions[corner_id] = pos

        # print self.corner_positions
        # print movement
        if (movement > 20 and len(detections) != 4) or len(detections) < 2:
            return
        pts_src = numpy.array(self.corner_locations)

        h, status = cv2.findHomography(pts_src, self.pts_dst)

        image = draw_detections_on_image(detections, image)
        trans_image = cv2.warpPerspective(image, h, self.output_shape)

        self.transformed_workspace_pub.publish(self.bridge.cv2_to_imgmsg(trans_image))
        # Should remove the white background. Will need to do this better
        # trans_image[numpy.sum(trans_image, axis=2) > 300] = 0
        if DEBUG:
            show_image(trans_image)

        if self.camera_params and not numpy.any(numpy.isnan(self.corner_positions)):

            centered_1 = self.corner_positions - numpy.mean(self.corner_positions, axis=0)
            centered_2 = self.corner_coords - numpy.mean(self.corner_coords, axis=0)

            C = numpy.dot(centered_1.T, centered_2) / self.corner_coords.shape[0]

            V, S, W = numpy.linalg.svd(C)
            d = (numpy.linalg.det(V) * numpy.linalg.det(W)) < 0.0
            if d:
                S[-1] = -S[-1]
                V[:, -1] = -V[:, -1]

            R = numpy.dot(V, W)
            trans = geometry_msgs.msg.TransformStamped()
            t = self.corner_positions.mean(axis=0) - numpy.dot(R, self.corner_coords.mean(axis=0).T)
            trans.header.stamp = rospy.Time.now()
            trans.header.frame_id = "camera_color_optical_frame"
            trans.child_frame_id = "workspace"
            trans.transform.translation.x = t[0]
            trans.transform.translation.y = t[1]
            trans.transform.translation.z = t[2]

            q = tf_conversions.transformations.quaternion_from_matrix(numpy.vstack([numpy.hstack([R, t.reshape((-1, 1))]), numpy.array([0,0,0,1.0])]))
            trans.transform.rotation.x = q[0]
            trans.transform.rotation.y = q[1]
            trans.transform.rotation.z = q[2]
            trans.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(trans)


            
            
            if DEBUG:
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera_color_optical_frame"
                
                pc = pcl2.create_cloud_xyz32(header, self.corner_positions)
                self.pc_pub.publish(pc)

                trans_points = numpy.dot(R, self.corner_coords.T).T + t
                pc = pcl2.create_cloud_xyz32(header, trans_points)
                self.pc_pub2.publish(pc)

    def new_camera_info(self, data):
        fx, _, cx, _, fy, cy, _, _, _ = data.K
        self.camera_params = (fx, fy, cx, cy)


if __name__ == '__main__':
    rospy.init_node('workspace_tracker', anonymous=True)
    wt = WorkspaceTracker()
    rospy.Subscriber("/camera/color/image_rect_color", Image, wt.new_image_message)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo, wt.new_camera_info)
    rospy.spin()