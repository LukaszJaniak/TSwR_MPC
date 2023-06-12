import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from math import sin, cos, tan, atan2, sqrt
from scipy.linalg import inv
from tf_transformations import euler_from_quaternion
from autoware_auto_control_msgs.msg import AckermannControlCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy


class linear_MPC(Node):
    def __init__(self):
        super().__init__('linear_MPC_ctrl')
        self.curr_pose = self.create_subscription(PoseStamped, '/ground_truth/pose',
                                                                  self.get_curr_pose, 10)
        self.sub_ref_traj = self.create_subscription(Path, '/path_points', self.get_ref_traj,10)
        qos_policy = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,depth=10)
        self.publish_control = self.create_publisher(AckermannControlCommand, '/control/command/control_cmd',
                                                       qos_policy)
        
        ctrl_pub_period = 1/30#0.033#0.03  # okres Musi byc >1/26hz -> odswiezanie ground_truth/pose
        self.control_publisher_timer = self.create_timer(ctrl_pub_period, self.periodic_publish_ctrl)
        
        update_period = 0.1 
        self.control_timer = self.create_timer(update_period, self.update_control)

        self.x, self.y, self.delta, self.prev_delta = None, None, None, None
        self.qx, self.qy, self.qz, self.qw = None, None, None, None
        self.v, self.ang_v, self.a = None, None, None
        self.ref_path = None
        self.yaw = None
        
        self.L = 0.45
        self.prev_vel_timestamp = self.get_clock().now()


    def get_curr_pose(self, msg: PoseStamped):
        self.x, self.y= msg.pose.position.x, msg.pose.position.y
        self.qx, self.qy, self.qz, self.qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
    
    
    def get_ref_traj(self, msg: Path):
        self.ref_path = []
        for pose in msg.poses:
            self.ref_path.append([pose.pose.position.x, pose.pose.position.y,
                                   euler_from_quaternion([pose.pose.orientation.x,
                                                          pose.pose.orientation.y,
                                                          pose.pose.orientation.z,
                                                          pose.pose.orientation.w])[2]])


    def periodic_publish_ctrl(self):
        if not None in [self.delta, self.a]:
            print("Kubica simulator: lekka kontra ", self.delta, " stopni i moidu na poziomie ", self.a)
            acc = AckermannControlCommand()
            acc.longitudinal.speed = 1.0
            acc.lateral.steering_tire_angle = self.delta
            acc.longitudinal.acceleration = self.a
            self.publish_control.publish(acc)


    def get_curr_speed(self):
        if len(self.ref_path) > 1:
            dx = self.ref_path[1][0] - self.ref_path[0][0]
            dy = self.ref_path[1][1] - self.ref_path[0][1]
            self.ref_yaw = atan2(dy, dx)
            dt = (self.get_clock().now().nanoseconds- self.prev_vel_timestamp.nanoseconds)/10**9
            self.v = sqrt(dx ** 2 + dy ** 2) / dt
            self.ang_v = (self.ref_yaw - self.yaw) / dt
        else:
            self.v = 0
            self.ang_v = 0
        self.prev_vel_timestamp = self.get_clock().now()


    def get_matrices_ABC(self, dt):
        A = np.eye(4)
        B = np.zeros((4, 2))
        C = np.zeros((4,1))
        if not None in [self.delta, self.a]:        
            A = np.array([[1, 0, cos(self.yaw)*dt, -self.v * sin(self.yaw)*dt],
                        [0, 1, sin(self.yaw)*dt, self.v * cos(self.yaw)*dt],
                        [0, 0, 1, 0],
                        [0, 0, (dt * tan(self.delta)) / self.L, 1]])
            
            B[2,0] = dt
            B[3,1] = dt * self.v / (self.L * cos(self.delta) ** 2)

            C[0,0] = self.v *sin(self.yaw) * self.yaw * dt
            C[1, 0] = - self.v * np.cos(self.yaw) * self.yaw * dt
            C[3, 0] = - self.v * self.delta * dt / (self.L * np.cos(self.delta)**2)
            C*=dt            
        return A, B, C 

    #Ograniczenie maksymalnego kątu skrętu kół/cykl
    def soft_turn(self, new_ang, max_angle):
        new_ang +=1.0
        prev_ang = self.delta +1.0
        if abs(new_ang-prev_ang) >max_angle:
            if new_ang > prev_ang:
                new_ang = prev_ang + max_angle -1.0
            else:
                new_ang = prev_ang - max_angle -1.0
        self.delta = new_ang

    #Sprawdzenie nasycenia wartości sterujących
    def check_bounds(self, bound_delta, bound_accel):
        if self.delta> bound_delta:
            self.delta = bound_delta
        elif self.delta <-bound_delta:
            self.delta = -bound_delta
        if self.a > bound_accel:
            self.a = bound_accel

    def update_control(self):
        if None in [self.x, self.y, self.qw, self.qx, self.qy, self.qz, self.ref_path]:
            return

        self.yaw = euler_from_quaternion([self.qx, self.qy, self.qz, self.qw])[2]
        dt = 0.1
        bound_delta = 1.0
        bound_accel = 0.5

        self.get_curr_speed()
        A, B, C = self.get_matrices_ABC(dt)
        Q = np.diag([1.0, 1.0, 0.5, 0.5]) # waga stanu
        R = np.diag([7500, 0.01])  # R waga sygnalu wejsciowego
        P = Q
        iter = 170
        eps = 0.01

        for _ in range(iter):
            Pn = Q + A.transpose() @ P @ A - A.transpose() @ P @ B @ inv(R + B.transpose() @ B) @ B.transpose() @ P @ A
            if (abs(Pn - P)).max() < eps:
                break
            P = Pn
        K = -inv(R + B.transpose() @ P @ B) @ B.transpose() @ P @ A
        cp_x, cp_y, cp_yaw = self.ref_path[0]
        e = np.array([self.x - cp_x, self.y - cp_y, self.v, self.yaw - cp_yaw])
        a, delta = K @ e
        self.a = -a
        if self.delta is not None:
            self.soft_turn(delta, 0.3) 
        self.delta = delta
        self.check_bounds(bound_delta, bound_accel)
        
        


def main(args=None):
    rclpy.init(args=args)
    MPC = linear_MPC()
    rclpy.spin(MPC)
    MPC.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


#Safe space 
# Calculate P cost-to-go matrix for DARE
        # Q = np.diag([1, 1, 0.5, 0.5]) # Q - weights on the state
        # R = np.diag([5000, 0.01])  # R - weights on control input. Large R if there is a limit on control output signal
        # P = Q
        # maxiter = 150
        # eps = 0.01

        # for i in range(maxiter):
        #     Pn = Q + A.T @ P @ A - A.T @ P @ B @ inv(R + B.T @ B) @ B.T @ P @ A
        #     if (abs(Pn - P)).max() < eps:
        #         break
        #     P = Pn

        # # Calculate feedback gain for DARE
        # K = -inv(R + B.T @ P @ B) @ B.T @ P @ A
        # x_error = np.array([self.x - self.ref_path[0][0],
        #                     self.y - self.ref_path[0][1],
        #                     self.v,
        #                     self.yaw - self.ref_path[0][2]])

        # u_optimal = K @ x_error
        # ##########Co tam się dzieje ^^^

        # if self.delta is not None:
        #     # self.soft_turn(u_optimal[1], 0.3) 
        #     self.delta = u_optimal[1]
        # else:
        #     self.delta = u_optimal[1]
        # self.a = -u_optimal[0]



    # def get_constraints(self, A, B, C):
    #     z = cp.Variable((4, len(self.ref_path + 1)))
    #     u = cp.Variable((2, len(self.ref_path)))

    #     constraints += [z[:, i+1] == A @ z[:, i] + B @ u[:, i] + C.flatten()]

    #         # Velocity limits
    #         constraints += [z[2, i] <= self.v_max]
    #         constraints += [z[2, i] >= self.v_min]

    #         # Input limits
    #         constraints += [self.a_min <= u[0, i]]
    #         constraints += [u[0, i] <= self.a_max]
    #         constraints += [u[1, i] <= self.max_steering_angle]
    #         constraints += [u[1, i] >= -self.max_steering_angle]
    #         # Rate of change of input limit
    #         if i != 0:
    #             constraints += [u[0, i] - u[0, i-1] <= self.a_rate_max]
    #             constraints += [u[0, i] - u[0, i-1] >= -self.a_rate_max]
    #             constraints += [u[1, i] - u[1, i-1] <= self.steer_rate_max * dt]
    #             constraints += [u[1, i] - u[1, i-1] >= -self.steer_rate_max * dt]
    #             constraints += [(z[0, i + 1] - z_ref[0, i])*np.sin(z_ref[3,i]) <= self.dist]
    #             constraints += [(z[0, i + 1] - z_ref[0, i])*np.sin(z_ref[3,i]) >= -self.dist]
    #             constraints += [(z[1, i + 1] - z_ref[1, i])*np.cos(z_ref[3,i]) <= self.dist]
    #             constraints += [(z[1, i + 1] - z_ref[1, i])*np.cos(z_ref[3,i]) >= -self.dist]
    #     return constraints

        ##########Smietnik
        # #TODO
        # Q = np.eye([2.5, 2.5, 1.1, 5.5])
        # Qf = np.eye([2.5, 2.5, 1.5, 3.5])
        # R = np.eye(2)
        # z_ref = np.array([[self.x], [self.y], [self.v], [self.yaw]])
        # z = cp.Variable((4, len(self.ref_path) + 1))
        # u = cp.Variable((2, len(self.ref_path)))
        # cost = 0
        # #constraints
        # # constraints = self.get_constraints()

        # # Terminal cost
        # cost += cp.quad_form(z_ref[:, -1] - z[:, -1], self.Qf)
        # # qp = cp.Problem(cp.Minimize(cost), constraints)
        # qp = cp.Problem(cp.Minimize(cost))
        # qp.solve(solver=cp.ECOS, verbose=False)

        # if qp.status == cp.OPTIMAL or qp.status == cp.OPTIMAL_INACCURATE:
        #     print("if works")
        #     self.a = np.array(u.value[0, :]).flatten()
        #     self.delta = np.array(u.value[1, :]).flatten()
        
        
        # self.prev_delta = self.delta