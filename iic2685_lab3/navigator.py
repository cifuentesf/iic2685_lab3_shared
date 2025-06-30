#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        self.step_distance = 0.15
        self.linear_speed = 0.12
        self.confidence_threshold = 0.8
        self.desired_wall_distance = 0.45
        self.collision_distance = 0.3
        self.front_collision_distance = 0.35
        self.kp = 1.2
        self.ki = 0.01
        self.kd = 0.08
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_time = None
        self.state = "filtering"
        self.current_scan = None
        self.localization_confidence = 0.0
        self.filter_iterations = 0
        self.max_filter_iterations = 25
        self.step_count = 0
        self.localized_announced = False
        self.movement_start_time = None
        self.movement_duration = 0.0
        self.is_moving = False
        self.left_distance = float('inf')
        self.left_front_distance = float('inf')
        self.front_distance = float('inf')
        self.right_front_distance = float('inf')
        self.right_distance = float('inf')
        self.left_wall_distance = float('inf')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.coef_sub = self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        self.best_pose_sub = self.create_subscription(PointStamped, '/best_pose', self.best_pose_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Navegador iniciado - Fase de filtrado inicial")

    def laser_callback(self, msg):
        self.current_scan = msg
        self.process_scan_data(msg)

    def process_scan_data(self, scan):
        if scan is None:
            return
        ranges = np.array(scan.ranges)
        ranges[~np.isfinite(ranges)] = 4.0
        ranges[ranges <= 0.0] = 4.0
        ranges[ranges > 3.9] = 4.0
        n = len(ranges)
        if n == 0:
            return
        sector_size = n // 5
        sectors = {
            'left': ranges[4*sector_size:],
            'left_front': ranges[3*sector_size:4*sector_size],
            'front': ranges[2*sector_size:3*sector_size],
            'right_front': ranges[sector_size:2*sector_size],
            'right': ranges[:sector_size]
        }
        for name, sector_data in sectors.items():
            valid_data = sector_data[sector_data < 3.9]
            if len(valid_data) > 0:
                distance = np.percentile(valid_data, 25)
            else:
                distance = float('inf')
            if name == 'left':
                self.left_distance = distance
                self.left_wall_distance = distance
            elif name == 'left_front':
                self.left_front_distance = distance
            elif name == 'front':
                self.front_distance = distance
            elif name == 'right_front':
                self.right_front_distance = distance
            elif name == 'right':
                self.right_distance = distance

    def confidence_callback(self, msg):
        self.localization_confidence = msg.data
        if self.localization_confidence >= self.confidence_threshold:
            if self.state != "localized" and not self.localized_announced:
                best_pose = self.estimate_robot_pose()
                self.get_logger().info(
                    f"¡ROBOT LOCALIZADO! Pose estimada: x={best_pose[0]:.3f}, y={best_pose[1]:.3f}, θ={best_pose[2]:.3f}"
                )
                self.localized_announced = True
                self.state = "localized"
        elif self.state == "filtering" and self.filter_iterations >= self.max_filter_iterations:
            self.state = "exploration"
            self.get_logger().info("Iniciando exploración reactiva")

    def best_pose_callback(self, msg):
        self.best_pose = msg

    def estimate_robot_pose(self):
        x = self.best_pose.linear.x if hasattr(self, 'best_pose') else 0.0
        y = self.best_pose.linear.y if hasattr(self, 'best_pose') else 0.0
        theta = self.best_pose.angular.z if hasattr(self, 'best_pose') else 0.0
        return (x, y, theta)

    def control_loop(self):
        if self.current_scan is None:
            return
        cmd = Twist()
        if self.state == "filtering":
            self.filtering_behavior()
        elif self.state == "exploration":
            self.exploration_behavior()
        elif self.state == "localized":
            self.continuous_navigation()

    def filtering_behavior(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.filter_iterations += 1

    def exploration_behavior(self):
        current_time = self.get_clock().now()
        if not self.is_moving:
            self.decide_next_movement()
            self.movement_start_time = current_time
            self.is_moving = True
        else:
            elapsed = (current_time - self.movement_start_time).nanoseconds * 1e-9
            if elapsed < self.movement_duration:
                self.execute_current_movement()
            else:
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.is_moving = False
                self.step_count += 1
                self.get_logger().info(
                    f"Paso {self.step_count} completado (confianza: {self.localization_confidence:.3f})"
                )

    def decide_next_movement(self):
        if self.front_distance < self.front_collision_distance:
            if self.left_distance > self.right_distance:
                self.movement_type = "turn_left"
                self.movement_duration = 1.5
            else:
                self.movement_type = "turn_right"
                self.movement_duration = 1.5
        else:
            self.movement_type = "forward"
            self.movement_duration = self.step_distance / self.linear_speed

    def execute_current_movement(self):
        cmd = Twist()
        if self.movement_type == "forward":
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif self.movement_type == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6
        elif self.movement_type == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = -0.6
        self.cmd_pub.publish(cmd)

    def continuous_navigation(self):
        if self.current_scan is None:
            return
        cmd = Twist()
        front_blocked = self.front_distance < self.front_collision_distance
        if front_blocked:
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            if left_space > right_space + 0.1:
                cmd.angular.z = 0.5
                cmd.linear.x = 0.05
            elif right_space > left_space + 0.1:
                cmd.angular.z = -0.5
                cmd.linear.x = 0.05
            else:
                cmd.angular.z = 0.6
                cmd.linear.x = 0.0
        elif self.left_distance < 1.2:
            current_time = self.get_clock().now()
            if self.last_time is not None:
                dt = (current_time - self.last_time).nanoseconds * 1e-9
                if dt > 0:
                    wall_error = self.left_distance - self.desired_wall_distance
                    if self.left_front_distance < self.desired_wall_distance * 0.8:
                        wall_error += (self.desired_wall_distance * 0.8 - self.left_front_distance) * 0.5
                    p_term = self.kp * wall_error
                    self.integral_error += wall_error * dt
                    i_term = self.ki * self.integral_error
                    d_term = self.kd * (wall_error - self.previous_error) / dt
                    angular_z = np.clip(p_term + i_term + d_term, -0.5, 0.5)
                    cmd.linear.x = self.linear_speed
                    cmd.angular.z = angular_z
                    self.previous_error = wall_error
            self.last_time = current_time
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.3
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()