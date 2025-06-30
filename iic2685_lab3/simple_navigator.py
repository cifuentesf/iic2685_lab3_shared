#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

class Simple_Navigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')

        # Parámetros de navegación
        self.step_distance = 0.15
        self.linear_speed = 0.1
        self.rotate_speed = 1.0
        self.time_advance = self.step_distance / self.linear_speed
        
        # Seguimiento de muro
        self.desired_wall_distance = 0.4
        self.wall_following_active = True  
        
        # Control PID
        self.Kp = 1.0
        self.Ki = 0.01
        self.Kd = 0.1
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

        # Estado
        self.state = "filtering"
        self.current_scan = None
        self.filter_iterations = 0
        self.max_filter_iterations = 20
        self.corner_detected = False

        # Sensores
        self.distancia_minima_colision = 0.4
        self.distancia_muro = None
        self.left_wall_distance = float('inf')
        self.localization_confidence = 0.0
        self.confidence_threshold = 0.8

        # Mensaje de velocidad
        self.vel_msg = Twist()

        # Subs y Pubs
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.confidence_sub = self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Navegador mejorado iniciado")

    def odom_callback(self, msg):
        pass  # Mantenemos por compatibilidad

    def laser_callback(self, msg):
        self.current_scan = msg
        self.split_scan_data(msg)

    def confidence_callback(self, msg):
        self.localization_confidence = msg.data
        if self.localization_confidence >= self.confidence_threshold:
            self.get_logger().info(f"¡Objetivo alcanzado! Confianza: {self.localization_confidence:.2f}")
            self.state = "localized"
            self.stop_robot()

    def control_loop(self):
        if self.state == "localized":
            return
            
        self.update_scan_data()
        self.detect_corner()
        
        if self.state == "filtering":
            self.ask_filter()
        elif self.state == "moving":
            self.move_robot()

    def detect_corner(self):
        """Detecta si está en una esquina (pared frontal e izquierda cerca)"""
        front_threshold = self.distancia_minima_colision * 1.5
        side_threshold = self.desired_wall_distance * 1.2
        
        self.corner_detected = (self.distancia_muro < front_threshold and 
                               self.left_wall_distance < side_threshold)
        
        if self.corner_detected:
            self.get_logger().warn("¡Esquina detectada! Girando para salir...")

    def ask_filter(self):
        self.filter_iterations += 1
        self.get_logger().info(f"Filtrado ({self.filter_iterations}/{self.max_filter_iterations})")
        
        if self.filter_iterations >= self.max_filter_iterations:
            self.state = "moving"
            self.filter_iterations = 0

    def move_robot(self):
        if self.corner_detected:
            # Comportamiento especial para salir de esquinas
            self.escape_corner()
        elif self.wall_collision():
            self.avoid_frontal_collision()
        else:
            self.follow_left_wall()

    def escape_corner(self):
        """Gira a la derecha para salir de esquinas"""
        self.vel_msg.linear.x = 0.05  # Avance lento
        self.vel_msg.angular.z = -0.5  # Giro moderado a derecha
        self.cmd_pub.publish(self.vel_msg)
        
    def avoid_frontal_collision(self):
        """Evita choques frontales"""
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = -0.3  # Giro suave a derecha
        self.cmd_pub.publish(self.vel_msg)

    def follow_left_wall(self):
        """Seguimiento de pared con PID"""
        if self.left_wall_distance == float('inf'):
            # Buscar pared si no se detecta
            self.vel_msg.linear.x = self.linear_speed
            self.vel_msg.angular.z = 0.3  # Giro suave a izquierda
            self.cmd_pub.publish(self.vel_msg)
            return
            
        # Cálculo PID
        error = self.left_wall_distance - self.desired_wall_distance
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        
        P = self.Kp * error
        self.integral += error * dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error)/dt if dt > 0 else 0
        
        angular_z = np.clip(P + I + D, -0.5, 0.5)
        
        self.vel_msg.linear.x = self.linear_speed
        self.vel_msg.angular.z = angular_z
        self.cmd_pub.publish(self.vel_msg)
        
        self.prev_error = error
        self.last_time = current_time

    def stop_robot(self):
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        self.cmd_pub.publish(self.vel_msg)

    def split_scan_data(self, scan_data):
        if scan_data is None:
            return
            
        ranges = np.array(scan_data.ranges)
        ranges[~np.isfinite(ranges)] = 4.0
        ranges[ranges <= 0.0] = 4.0

        # Regiones de interés
        num_sectors = 6
        sector_size = len(ranges) // num_sectors
        self.scan_front = ranges[2*sector_size:3*sector_size]
        self.scan_far_left = ranges[5*sector_size:]

    def update_scan_data(self):
        if self.current_scan is None:
            return
            
        # Distancia frontal (percentil 25 para evitar outliers)
        valid_front = np.array(self.scan_front)
        valid_front = valid_front[valid_front < 3.9]
        self.distancia_muro = np.percentile(valid_front, 25) if len(valid_front) > 0 else float('inf')
        
        # Distancia izquierda (percentil 50 para estabilidad)
        valid_left = np.array(self.scan_far_left)
        valid_left = valid_left[valid_left < 3.9]
        self.left_wall_distance = np.percentile(valid_left, 50) if len(valid_left) > 0 else float('inf')

    def wall_collision(self):
        return self.distancia_muro < self.distancia_minima_colision

def main(args=None):
    rclpy.init(args=args)
    navigator = Simple_Navigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()