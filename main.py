#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
import cv2
import numpy as np
from openvino_models import *
import math
from nav_msgs.msg import Odometry
import actionlib
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, PointStamped, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from actionlib_msgs.msg import GoalStatusArray
import time

def point_to_pose(x, y, theta):
    q = quaternion_from_euler(0.0, 0.0, theta)
    location = Pose(Point(x, y, 0.0), Quaternion(q[0], q[1], q[2], q[3]))
    return location

def callback_pose(msg):
	global pose
	pose = msg.pose.pose

def callback_img(msg):
	global frame
	frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global depth
    tmp = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    depth = np.array(tmp, dtype=np.float32)
    
def move_to(x, y, theta):
    move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    location = point_to_pose(x, y, theta)
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = location
    move_base.send_goal(goal)
    time.sleep(1)
    
if __name__ == "__main__":
    rospy.init_node("TISF_2023")
        
    frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_img)
        
    depth = None
    topic_name = "/camera/depth/image_raw"
    rospy.Subscriber(topic_name, Image, callback_depth)
    rospy.wait_for_message(topic_name, Image)
        
    pose = None
    rospy.Subscriber("/odom", Odometry, callback_pose)
    rospy.sleep(1)
        
    print(pose)
        
    dnn_face = FaceDetection()
        
    print("Entering Main Loop")
        
    status = 0
    xf, yf = 0, 0
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if frame is None:
            print("Frame is None")
            continue
        if depth is None: 
            print("Depth is None")
            continue
        if pose is None: 
            print("Pose is None")
            continue
        
		
        robot_orientation = math.degrees(pose.orientation.z)
                
        gray = depth / np.max(depth)
		
        boxes = dnn_face.forward(frame)
                
        for x1, y1, x2, y2 in boxes:
            faces = frame[y1:y2, x1:x2, :].copy()

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
            depth_to_person = depth[cy][cx] // 5
            x = cx / 480 * 2 * depth_to_person * math.tan(math.radians(60) / 2)
            AM = x

            AC = (depth_to_person** 2 + AM ** 2) ** 0.5
			
            angle_ACM = math.degrees(math.atan(AM/depth_to_person))
            angle_2 = 90 - robot_orientation - angle_ACM
            #print(robot_orientation)
            y_final = math.sin(math.radians(angle_2)) * AC * 0.05
            x_final = math.cos(math.radians(angle_2)) * AC * 0.05

            print("coordinates: %.4f, %.4f" % (x_final / 100, y_final / 100))

            xf, yf = x_final / 100, y_final / 100
            if xf != 0 and yf != 0:
                status = 1
                continue
			
        if status == 1:
            robot_position_x = pose.position.x
            robot_position_y = pose.position.y
            move_to(robot_position_x + xf, robot_position_y + yf, -0.111234)		
            break
			
        cv2.imshow("gray", gray)
        cv2.imshow("frame", frame)	

        cv2.waitKey(1)
		
