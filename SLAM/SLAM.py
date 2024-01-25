# Student name: 

import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import matplotlib.pyplot as plt
import time
from tf2_msgs.msg import TFMessage
from copy import copy
from visualization_msgs.msg import Marker

# Further info:
# On markers: http://wiki.ros.org/rviz/DisplayTypes/Marker
# Laser Scan message: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

class CodingExercise3(Node):

    def __init__(self):
        super().__init__('CodingExercise3')

        self.ranges = [] # lidar measurements
        
        self.point_list = [] # A list of points to draw lines
        self.line = Marker()
        self.line_marker_init(self.line)


        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'terrasentia/scan', self.callback_scan, 10)
        self.pub_lines = self.create_publisher(Marker, 'lines', 10)
        self.timer_draw_line_example = self.create_timer(0.1, self.draw_line_example_callback)
        self.current_pose = Pose()
        self.line_marker_init(self.line)  

    def callback_ekf(self, msg):
        # You will need this function to read the translation and rotation of the robot with respect to the odometry frame
        self.current_pose = msg.pose.pose

    def callback_scan(self, msg):
        self.ranges = list(msg.ranges)  # Lidar measurements
        # print(self.ranges)
        # self.pub_lines.publish(self.ranges)
        min_range = 1.25  # Set your minimum range threshold here
        max_range = 20.0  # Set your maximum range threshold here
        filtered_ranges = [range_val if min_range < range_val < max_range else 0.0 for range_val in msg.ranges]
        
        # Process the filtered lidar data
        self.process_lidar_data(filtered_ranges)





    def polar_to_cartesian(self,rho, theta):
        x = rho * np.sin(theta)
        y = rho * np.cos(theta)
        return x, y
    
    def point_line_distance(self,x, y, x1, y1, x2, y2):
        absolute_dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        normalization = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return absolute_dist / normalization

    def split_and_merge(self,points, start, end, threshold):
        # Break case
        if start >= end:
            return []
        
        x1, y1 = points[start]
        x2, y2 = points[end]
        max_dist = 0
        farthest_index = start

        # Loop through points and find farthest points
        for i in range(start + 1, end):
            x, y = points[i]
            dist = self.point_line_distance(x, y, x1, y1, x2, y2)
            if dist > max_dist:
                max_dist = dist
                farthest_index = i

        # Recurse down
        if max_dist > threshold:
            # Split step
            half1 = self.split_and_merge(points, start, farthest_index, threshold)
            half2 = self.split_and_merge(points, farthest_index, end, threshold)

            # Merge step
            return half1 + half2
        else:
            # Leaf Smallest Entry
            return [(start, end)]
    
    def transform_to_global_frame(self, local_points, pose):
        position = pose.position
        orientation = pose.orientation

        # Convert quaternion to rotation matrix
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = self.quaternion_to_rotation_matrix(q)

        global_points = []
        for point in local_points:
            # Apply rotation
            rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1], 0]))

            # Apply translation
            transformed_point = rotated_point + np.array([position.x, position.y, position.z])
            global_points.append(transformed_point[:2])  # We only need x, y for 2D mapping

        return np.array(global_points)

    def quaternion_to_rotation_matrix(self, q):
        # Extract the values from q
        x, y, z, w = q

        # Compute the rotation matrix elements
        r00 = 1 - 2 * y * y - 2 * z * z
        r01 = 2 * x * y - 2 * z * w
        r02 = 2 * x * z + 2 * y * w

        r10 = 2 * x * y + 2 * z * w
        r11 = 1 - 2 * x * x - 2 * z * z
        r12 = 2 * y * z - 2 * x * w

        r20 = 2 * x * z - 2 * y * w
        r21 = 2 * y * z + 2 * x * w
        r22 = 1 - 2 * x * x - 2 * y * y

        # Construct the rotation matrix
        rotation_matrix = np.array([[r00, r01, r02],
                                    [r10, r11, r12],
                                    [r20, r21, r22]])

        return rotation_matrix


    def process_lidar_data(self, filtered_ranges):
        # Convert LiDAR data to Cartesian coordinates

        angles = np.linspace(-45, 225, len(filtered_ranges), dtype=np.float32) * (math.pi / 180.0)

        # Remove zero values which are placeholders for the outliers
        valid_ranges = [r for r in filtered_ranges if r != 0.0]
        valid_angles = [angles[i] for i, r in enumerate(filtered_ranges) if r != 0.0]

        x, y = self.polar_to_cartesian(np.array(valid_ranges), np.array(valid_angles))

        
        # Transform LiDAR points to global frame
        points = np.column_stack((x, y))
        
        points = [point for point in points if point[0] < 20.0 and point[0] > 1.25]
        # print(points)
        global_points = self.transform_to_global_frame(points, self.current_pose)

        # Apply Split-and-Merge on global points
        start, end = 0, len(global_points) - 1
        line_segments = self.split_and_merge(global_points, start, end, threshold=3)
        # self.point_list=[]
        # gap_thres = 5.0
        # for i in range(len(line_segments)):
        #     # Extract the start and end points for the current segment
        #     start_idx, end_idx = line_segments[i]
        #     start_point = global_points[start_idx]
        #     end_point = global_points[end_idx]
            
        #     # Add the start point of the segment to the point list
        #     p0 = Point(x=start_point[0], y=start_point[1], z=0.0)
        #     self.point_list.append(copy(p0))
            
        #     # Check if we're not at the last segment and if there's a gap to the next segment
        #     if i < len(line_segments) - 1:
        #         next_start_point = points[line_segments[i+1][0]]
        #         print(np.linalg.norm(np.array(end_point) - np.array(next_start_point)))
        #         if np.linalg.norm(np.array(end_point) - np.array(next_start_point)) > gap_thres:
        #             # Gap detected, do not connect this segment to the next one
        #             # Add the end point of the current segment to the point list
        #             p1 = Point(x=end_point[0], y=end_point[1], z=0.0)
        #             self.point_list.append(copy(p1))
        #             continue
            
        #     # If there's no gap or it's the last segment, connect to the next start point
        #     # This ensures that the last segment end point is always added
        #     if i == len(line_segments) - 1:
        #         p1 = Point(x=end_point[0], y=end_point[1], z=0.0)
        #         self.point_list.append(copy(p1))

        # if len(self.point_list) % 2 != 0:
        #     # If there's an odd number of points, remove the last one
        #     self.point_list.pop()


        for start_idx, end_idx in line_segments:
            p0 = Point()
            p0.x = float(global_points[start_idx][0])
            p0.y = float(global_points[start_idx][1])
            p0.z = 0.0

            p1 = Point()
            p1.x = float(global_points[end_idx][0])
            p1.y = float(global_points[end_idx][1])
            p1.z = 0.0

            self.point_list.extend([copy(p0), copy(p1)])

        # Update the line marker points and publish
        self.line.points = self.point_list
        self.pub_lines.publish(self.line)


    def draw_line_example_callback(self):
        # Here is just a simple example on how to draw a line on rviz using line markers. Feel free to use any other method
        p0 = Point()
        p0.x = 0.0
        p0.y = 0.0
        p0.z = 0.0

        p1 = Point()
        p1.x = 1.0
        p1.y = 1.0
        p1.z = 1.0

        self.point_list.append(copy(p0)) 
        self.point_list.append(copy(p1)) # You can append more pairs of points
        self.line.points = self.point_list

        # self.pub_lines.publish(self.line) # It will draw a line between each pair of points

    def line_marker_init(self, line):
        line.header.frame_id="/odom"
        line.header.stamp=self.get_clock().now().to_msg()

        line.ns = "markers"
        line.id = 0

        self.line.id = 0
        self.line.type = Marker.LINE_LIST
        self.line.action = Marker.ADD
        self.line.pose.orientation.w = 1.0
        self.line.scale.x = 0.05
        self.line.scale.y = 0.05
        self.line.color.r = 1.0
        self.line.color.a = 1.0  
        # line.lifetime = 0
        


def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
