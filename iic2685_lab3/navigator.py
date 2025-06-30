#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        # Parámetros simplificados
        self.linear_speed = 0.25
        self.max_angular_speed = 0.8
        self.wall_distance = 0.35
        self.front_threshold = 0.4
        self.side_threshold = 0.25
        
        # Estado
        self.exploration_active = True
        self.current_scan = None
        self.state = "wall_following"  # Estados: wall_following, turning_right, turning_left
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicador
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer para control
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador reactivo iniciado")
        
    def laser_callback(self, msg):
        """Callback para datos LIDAR"""
        self.current_scan = msg
        
    def confidence_callback(self, msg):
        """Callback para confianza de localización"""
        if msg.data > 0.8 and self.exploration_active:
            self.exploration_active = False
            self.get_logger().info(f"Localización completada (confianza: {msg.data:.3f}). Deteniendo exploración.")
            
    def control_loop(self):
        """Loop principal de control"""
        if not self.exploration_active:
            self.stop_robot()
            return
            
        if self.current_scan is None:
            return
            
        self.reactive_navigation()
        
    def reactive_navigation(self):
        """Navegación reactiva simple"""
        ranges = np.array(self.current_scan.ranges)
        
        # Filtrar lecturas inválidas
        ranges[ranges >= self.current_scan.range_max] = self.current_scan.range_max
        ranges[ranges <= self.current_scan.range_min] = self.current_scan.range_max
        ranges[~np.isfinite(ranges)] = self.current_scan.range_max
        
        # Dividir escaneo en regiones
        n = len(ranges)
        front_ranges = ranges[n//3:2*n//3]
        right_ranges = ranges[2*n//3:]
        left_ranges = ranges[:n//3]
        
        # Distancias mínimas
        front_dist = np.min(front_ranges)
        right_dist = np.min(right_ranges)
        left_dist = np.min(left_ranges)
        
        cmd = Twist()
        
        # Lógica de navegación basada en estados
        if self.state == "wall_following":
            if front_dist < self.front_threshold:
                # Obstáculo adelante - decidir dirección
                if right_dist > left_dist:
                    self.state = "turning_right"
                else:
                    self.state = "turning_left"
            else:
                # Seguir pared derecha
                if right_dist > self.wall_distance * 1.5:
                    # No hay pared derecha - girar derecha para buscarla
                    cmd.linear.x = self.linear_speed * 0.7
                    cmd.angular.z = -0.5
                elif right_dist < self.side_threshold:
                    # Muy cerca de la pared - alejarse
                    cmd.linear.x = self.linear_speed * 0.8
                    cmd.angular.z = 0.3
                else:
                    # Seguir adelante con corrección proporcional
                    error = right_dist - self.wall_distance
                    cmd.linear.x = self.linear_speed
                    cmd.angular.z = -0.8 * error  # Control proporcional
                    
        elif self.state == "turning_right":
            cmd.angular.z = -self.max_angular_speed
            cmd.linear.x = 0.0
            # Volver a seguimiento cuando hay espacio adelante
            if front_dist > self.front_threshold * 1.2:
                self.state = "wall_following"
                
        elif self.state == "turning_left":
            cmd.angular.z = self.max_angular_speed
            cmd.linear.x = 0.0
            # Volver a seguimiento cuando hay espacio adelante
            if front_dist > self.front_threshold * 1.2:
                self.state = "wall_following"
        
        # Limitar velocidades
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)
        
        self.cmd_pub.publish(cmd)
        
    def stop_robot(self):
        """Detener el robot"""
        cmd = Twist()
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