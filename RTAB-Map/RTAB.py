import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')
        self.rtabmap_subscriber = self.create_subscription(
            Odometry, '/rtabmap/odom', self.rtabmap_callback, 10)
        self.ekf_subscriber = self.create_subscription(
            Odometry, '/terrasentia/ekf', self.ekf_callback, 10)

        self.rtabmap_data = []
        self.ekf_data = []

    def rtabmap_callback(self, data):
        position = data.pose.pose.position
        self.rtabmap_data.append((position.x, position.y, position.z))
        self.save_data('rtabmap_trajectory_b2.txt', self.rtabmap_data)

    def ekf_callback(self, data):
        position = data.pose.pose.position
        self.ekf_data.append((position.x, position.y, position.z))
        self.save_data('ekf_trajectory_b2.txt', self.ekf_data)

    def save_data(self, filename, data):
        with open(filename, 'a') as file:
            file.write(f"{data[-1][0]}, {data[-1][1]}, {data[-1][2]}\n")

# ROS2 Node Initialization
rclpy.init()
trajectory_node = TrajectoryNode()

# Spin the node
rclpy.spin(trajectory_node)

# Clean up and shut down
trajectory_node.destroy_node()
rclpy.shutdown()
