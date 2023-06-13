import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped 
from tf_transformations import euler_from_quaternion
from autoware_auto_control_msgs.msg import AckermannControlCommand 
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')
        self.current_pose_subscription = self.create_subscription(PoseStamped,'/ground_truth/pose', self.get_curr_pose, 10)
        self.x, self.y, self.z = None, None, None
        self.qx, self.qy, self.qz, self.qw = None, None, None, None

        self.reference_trajectory_subscription = self.create_subscription(Path, '/path_points', self.get_ref_traj, 10)
        self.ref_path = None
        qos_policy = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, depth=10)
        self.control_publisher = self.create_publisher(AckermannControlCommand, '/control/command/control_cmd', qos_policy)
        ctrl_pub_period = 0.02
        self.control_publisher_timer = self.create_timer(ctrl_pub_period, self.periodic_publish_ctrl)
        control_timer = 0.1 
        self.control_timer = self.create_timer(control_timer, self.update_control)
        self.delta,self.a = None, None
        self.wheelbase = 0.4 
        self.track = 0.23  
        self.lookahead_distance = 0.7  


    def get_curr_pose(self, msg:PoseStamped):
        self.x, self.y= msg.pose.position.x, msg.pose.position.y
        self.qx, self.qy, self.qz, self.qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w


    def get_ref_traj(self, msg:Path):
        self.ref_path = []
        for pose in msg.poses:
            self.ref_path.append([pose.pose.position.x, pose.pose.position.y,
                                   euler_from_quaternion([pose.pose.orientation.x,
                                                          pose.pose.orientation.y,
                                                          pose.pose.orientation.z,
                                                          pose.pose.orientation.w])[2]])


    def publish_control(self, delta, accel):
        acc = AckermannControlCommand()
        acc.longitudinal.speed = 1.0
        acc.lateral.steering_tire_angle = delta
        acc.longitudinal.acceleration = accel
        self.control_publisher.publish(acc)


    def periodic_publish_ctrl(self):
        if not None in [self.delta, self.a]:
            self.publish_control(self.delta, self.a)
            print(f'delta: {self.delta}, acceleration: {self.a}')


    def update_control(self):
        if None in [self.x, self.y, self.qw, self.qx, self.qy, self.qz, self.ref_path]:
            return

        curr_pose = np.array([self.x, self.y])
        curr_yaw = euler_from_quaternion([self.qx, self.qy, self.qz, self.qw])[2]

        ref_points = np.array([[p[0], p[1]] for p in self.ref_path])

        distances = np.linalg.norm(ref_points - curr_pose, axis=1)

        nearest_idx = np.argmin(distances)


        for i in range(nearest_idx, len(self.ref_path)):
            ref_point = np.array([self.ref_path[i][0], self.ref_path[i][1]])
            if np.linalg.norm(ref_point - curr_pose) > self.lookahead_distance:
                lookahead_point = ref_point
                break
        else:
            lookahead_point = ref_points[-1]

        dx = lookahead_point[0] - curr_pose[0]
        dy = lookahead_point[1] - curr_pose[1]
        alpha = np.arctan2(dy, dx) - curr_yaw
        Lf = self.lookahead_distance
        steering_angle = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), Lf)
        self.delta = steering_angle
        self.a = 0.08



def main(args=None):
    rclpy.init(args=args)
    PurePursuitController = PurePursuitNode()
    rclpy.spin(PurePursuitController)
    PurePursuitController.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
