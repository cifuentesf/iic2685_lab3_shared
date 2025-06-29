#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class Navigation(Node):
    
    def __init__(self):
        super().__init__('navigation')
        
        # Parámetros
        self.linear_speed = 0.3          # [m/s]
        self.max_angular_speed = 1.0     # [rad/s]
        self.desired_wall_distance = 0.3 # [m]
        self.min_front_distance = 0.3   # [m]
        
        # Estado de navegación
        self.exploration_active = True
        self.current_laser_scan = None
        self.turning = False
        
        # Suscriptores
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        self.confidence_subscriber = self.create_subscription(
            Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicadores
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer para control de navegación
        self.navigation_timer = self.create_timer(0.2, self.navigation_control_loop)
        
        self.get_logger().info("Navegación Reactiva inicializada")
        
    def laser_callback(self, msg):
        """Callback LIDAR"""
        self.current_laser_scan = msg
        
    def confidence_callback(self, msg):
        # Si la confianza es muy alta, detener exploración
        if msg.data > 0.8 and self.exploration_active:
            self.exploration_active = False
            self.get_logger().info(f"Alta confianza detectada ({msg.data:.3f}). Deteniendo exploración.")
        
    def navigation_control_loop(self):
        if not self.exploration_active or self.current_laser_scan is None:
            self.stop_robot()
            return

        self.reactive_wall_following()
        
    def reactive_wall_following(self):

        scan = self.current_laser_scan
        ranges = np.array(scan.ranges)
        
        # Filtrar lecturas inválidas
        valid_ranges = ranges.copy()
        valid_ranges[valid_ranges >= scan.range_max] = scan.range_max
        valid_ranges[valid_ranges <= scan.range_min] = scan.range_max
        
        # Dividir el escaneo en regiones
        n_rays = len(valid_ranges)
        third = n_rays // 3
        
        left_ranges = valid_ranges[:third]
        front_ranges = valid_ranges[third:2*third]
        right_ranges = valid_ranges[2*third:]
        
        # Distancias mínimas en cada región
        front_dist = np.min(front_ranges)
        left_dist = np.min(left_ranges)
        right_dist = np.min(right_ranges)
        
        cmd = Twist()
        
        if front_dist < self.min_front_distance:
            # Obstáculo de frente
            if not self.turning:
                self.turning = True
            
            # Rotar hacia el lado con más espacio
            if right_dist > left_dist:
                cmd.angular.z = -self.max_angular_speed  # Girar a la derecha
            else:
                cmd.angular.z = self.max_angular_speed   # Girar a la izquierda
            cmd.linear.x = 0.0
            
        else:
            # Camino libre
            if self.turning:
                self.turning = False
            
            # Seguir la pared derecha a distancia fija
            error = right_dist - self.desired_wall_distance
            
            # Control proporcional simple para seguimiento de pared
            cmd.linear.x = self.linear_speed
            cmd.angular.z = -0.5 * error  # e*Kp

            cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd.angular.z))
        
        self.cmd_vel_publisher.publish(cmd)
        
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigation()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()