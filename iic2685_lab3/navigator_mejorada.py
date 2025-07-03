#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped
import time

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        # Parámetros de navegación
        self.linear_speed = 0.15
        self.angular_speed = 0.5
        self.confidence_threshold = 0.8
        self.desired_wall_distance = 0.4
        self.collision_distance = 0.25
        self.scan_time = 2.0
        self.move_time = 1.5
        
        # Parámetros PID para seguimiento de pared
        self.kp = 2.0
        self.ki = 0.1
        self.kd = 0.5
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Estado del navegador
        self.state = "scanning"
        self.current_scan = None
        self.localization_confidence = 0.0
        self.localized_announced = False
        
        # Variables de tiempo
        self.phase_start_time = None
        self.is_in_phase = False
        
        # Regiones del LIDAR (5 sectores)
        self.regions = {
            'right': 0.0,
            'right_center': 0.0,
            'center': 0.0,
            'left_center': 0.0,
            'left': 0.0
        }
        
        # Variables para mejor pose
        self.best_pose = None
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float64, '/localization_confidence', self.confidence_callback, 10)
        self.best_pose_sub = self.create_subscription(
            PointStamped, '/best_pose', self.best_pose_callback, 10)
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer para control
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Navegador iniciado - Patron: Movimiento -> Escaneo -> Movimiento -> ...")

    def laser_callback(self, msg):
        self.current_scan = msg
        self.process_scan_regions(msg)

    def process_scan_regions(self, scan):
        if scan is None:
            return
        ranges = np.array(scan.ranges)
        ranges[~np.isfinite(ranges)] = 4.0
        ranges[ranges <= 0.0] = 4.0
        ranges[ranges > 4.0] = 4.0
        n = len(ranges)
        if n == 0:
            return
        sector_size = n // 5
        self.regions = {
            'right': np.min(ranges[0:sector_size]) if sector_size > 0 else 10.0,
            'right_center': np.min(ranges[sector_size:2*sector_size]) if sector_size > 0 else 10.0,
            'center': np.min(ranges[2*sector_size:3*sector_size]) if sector_size > 0 else 10.0,
            'left_center': np.min(ranges[3*sector_size:4*sector_size]) if sector_size > 0 else 10.0,
            'left': np.min(ranges[4*sector_size:]) if n > 4*sector_size else 10.0
        }

    def confidence_callback(self, msg):
        self.localization_confidence = msg.data
        if (self.localization_confidence > self.confidence_threshold and 
            not self.localized_announced):
            self.stop_robot()
            best_pose = self.estimate_robot_pose()
            self.get_logger().info(
                f"ROBOT LOCALIZADO!"
            )
            self.get_logger().info(
                f"Confianza: {self.localization_confidence:.3f}"
            )
            self.get_logger().info(
                f"Posicion estimada: x={best_pose[0]:.3f}m, y={best_pose[1]:.3f}m"
            )
            self.get_logger().info("Robot detenido.")
            self.localized_announced = True
            self.state = "localized"

    def best_pose_callback(self, msg):
        self.best_pose = msg

    def estimate_robot_pose(self):
        if self.best_pose is not None:
            x = self.best_pose.point.x
            y = self.best_pose.point.y
        else:
            x = y = 0.0
        return (x, y)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if self.current_scan is None:
            return
        if self.state == "localized":
            self.stop_robot()
            return
        current_time = self.get_clock().now()
        if not self.is_in_phase:
            self.phase_start_time = current_time
            self.is_in_phase = True
            if self.state == "scanning":
                self.get_logger().info(
                    f"Iniciando fase de ESCANEO - Confianza inicial: {self.localization_confidence:.3f}"
                )
            else:
                self.get_logger().info("Iniciando fase de MOVIMIENTO")
        elapsed_time = (current_time - self.phase_start_time).nanoseconds * 1e-9
        if self.state == "scanning":
            self.stop_robot()
            if hasattr(self, '_last_confidence_report'):
                if elapsed_time - self._last_confidence_report > 0.5:
                    self.get_logger().info(
                        f"Escaneando... Confianza: {self.localization_confidence:.3f}"
                    )
                    self._last_confidence_report = elapsed_time
            else:
                self._last_confidence_report = elapsed_time
            if elapsed_time >= self.scan_time:
                self.get_logger().info(
                    f"Escaneo completado - Confianza: {self.localization_confidence:.3f}"
                )
                self.state = "moving"
                self.is_in_phase = False
        elif self.state == "moving":
            if elapsed_time >= self.move_time:
                self.get_logger().info(
                    f"Movimiento completado - Confianza: {self.localization_confidence:.3f}"
                )
                self.state = "scanning"
                self.is_in_phase = False
                self.stop_robot()
            else:
                self.intelligent_navigation()

    def intelligent_navigation(self):
        cmd = Twist()
        if self.is_collision_imminent():
            self.get_logger().info("Evitando colision")
            cmd = self.avoid_collision()
        elif self.has_left_wall():
            cmd = self.follow_left_wall()
        else:
            self.get_logger().info("Buscando pared izquierda")
            cmd = self.search_left_wall()
        self.cmd_pub.publish(cmd)

    def is_collision_imminent(self):
        return (self.regions['center'] < self.collision_distance or
                self.regions['left_center'] < self.collision_distance or
                self.regions['right_center'] < self.collision_distance)

    def avoid_collision(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        left_space = (self.regions['left'] + self.regions['left_center']) / 2
        right_space = (self.regions['right'] + self.regions['right_center']) / 2
        if left_space > right_space:
            cmd.angular.z = self.angular_speed
            self.get_logger().info("Rotando a la izquierda")
        else:
            cmd.angular.z = -self.angular_speed
            self.get_logger().info("Rotando a la derecha")
        return cmd

    def has_left_wall(self):
        return self.regions['left'] < 3.0

    def follow_left_wall(self):
        cmd = Twist()
        if (self.regions['right'] < self.collision_distance and 
            abs(self.regions['left'] - self.desired_wall_distance) < 0.1):
            target_distance = (self.regions['left'] + self.regions['right']) / 2
            error = target_distance - self.regions['left']
            self.get_logger().info(f"Navegando entre paredes - Distancia objetivo: {target_distance:.2f}m")
        else:
            error = self.desired_wall_distance - self.regions['left']
        self.integral_error += error * 0.05
        derivative_error = error - self.previous_error
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        angular_velocity = (self.kp * error + 
                          self.ki * self.integral_error + 
                          self.kd * derivative_error)
        angular_velocity = max(-self.angular_speed, min(self.angular_speed, angular_velocity))
        cmd.linear.x = self.linear_speed
        cmd.angular.z = angular_velocity
        self.previous_error = error
        if hasattr(self, '_last_log_time'):
            if time.time() - self._last_log_time > 1.0:
                self.get_logger().info(
                    f"Siguiendo pared izq: {self.regions['left']:.2f}m "
                    f"(objetivo: {self.desired_wall_distance:.2f}m)"
                )
                self._last_log_time = time.time()
        else:
            self._last_log_time = time.time()
        return cmd

    def search_left_wall(self):
        cmd = Twist()
        cmd.linear.x = self.linear_speed * 0.5
        cmd.angular.z = self.angular_speed * 0.3
        return cmd


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