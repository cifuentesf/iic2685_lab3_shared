#!/usr/bin/env python3
"""
Nodo de seguimiento de pared para exploración durante localización
Basado en el comportamiento reactivo del laboratorio 2
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np
import math


class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        
        # Parámetros de control
        self.declare_parameter('wall_distance', 0.5)  # Distancia deseada a la pared
        self.declare_parameter('max_linear_vel', 0.2)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('kp_distance', 1.0)  # Ganancia proporcional para distancia
        self.declare_parameter('kp_angle', 2.0)     # Ganancia proporcional para ángulo
        
        self.wall_distance = self.get_parameter('wall_distance').get_parameter_value().double_value
        self.max_linear_vel = self.get_parameter('max_linear_vel').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_vel').get_parameter_value().double_value
        self.kp_distance = self.get_parameter('kp_distance').get_parameter_value().double_value
        self.kp_angle = self.get_parameter('kp_angle').get_parameter_value().double_value
        
        # Estado
        self.is_active = True
        self.last_scan = None
        self.following_side = 'right'  # 'left' o 'right'
        self.rotation_count = 0
        
        # Suscriptores
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        self.localized_sub = self.create_subscription(
            Bool,
            '/robot_localized',
            self.localized_callback,
            10
        )
        
        # Publicador
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel_mux/input/navigation',  # Usar cmd_vel_mux como en labs anteriores
            10
        )
        
        # Timer para control
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Nodo de seguimiento de pared iniciado')
    
    def laser_callback(self, msg):
        """Actualiza la última lectura del láser"""
        self.last_scan = msg
    
    def localized_callback(self, msg):
        """Detiene el seguimiento cuando el robot está localizado"""
        if msg.data:
            self.is_active = False
            # Detener robot
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info('Robot localizado, deteniendo exploración')
    
    def get_wall_distance(self, scan, angle_start, angle_end):
        """
        Calcula la distancia mínima a la pared en un rango de ángulos
        """
        if scan is None:
            return float('inf')
        
        # Convertir ángulos a índices
        start_idx = int((angle_start - scan.angle_min) / scan.angle_increment)
        end_idx = int((angle_end - scan.angle_min) / scan.angle_increment)
        
        # Asegurar índices válidos
        start_idx = max(0, min(start_idx, len(scan.ranges) - 1))
        end_idx = max(0, min(end_idx, len(scan.ranges) - 1))
        
        # Obtener distancias válidas
        distances = []
        for i in range(start_idx, end_idx + 1):
            if scan.ranges[i] > scan.range_min and scan.ranges[i] < scan.range_max:
                distances.append(scan.ranges[i])
        
        return min(distances) if distances else float('inf')
    
    def control_loop(self):
        """
        Loop principal de control para seguimiento de pared
        """
        if not self.is_active or self.last_scan is None:
            return
        
        twist = Twist()
        
        # Distancias en diferentes direcciones
        dist_front = self.get_wall_distance(self.last_scan, -0.5, 0.5)
        dist_left = self.get_wall_distance(self.last_scan, 1.0, 1.57)
        dist_right = self.get_wall_distance(self.last_scan, -1.57, -1.0)
        
        # Detectar obstáculo frontal
        if dist_front < 0.4:
            # Girar 90 grados
            twist.linear.x = 0.0
            twist.angular.z = self.max_angular_vel
            self.rotation_count += 1
            
            # Si hemos girado muchas veces, cambiar de lado
            if self.rotation_count > 4:
                self.following_side = 'left' if self.following_side == 'right' else 'right'
                self.rotation_count = 0
        else:
            self.rotation_count = 0
            
            # Seguimiento de pared
            if self.following_side == 'right':
                wall_dist = dist_right
                error_distance = self.wall_distance - wall_dist
                
                # Control proporcional
                if wall_dist < float('inf'):
                    # Velocidad lineal (más lento si estamos muy cerca)
                    twist.linear.x = self.max_linear_vel * min(1.0, wall_dist / self.wall_distance)
                    
                    # Velocidad angular para mantener distancia
                    twist.angular.z = self.kp_distance * error_distance
                    
                    # Limitar velocidad angular
                    twist.angular.z = max(-self.max_angular_vel, 
                                         min(self.max_angular_vel, twist.angular.z))
                else:
                    # No hay pared, girar para encontrarla
                    twist.linear.x = 0.1
                    twist.angular.z = -0.3
            else:
                # Seguimiento por la izquierda (similar pero con signos invertidos)
                wall_dist = dist_left
                error_distance = self.wall_distance - wall_dist
                
                if wall_dist < float('inf'):
                    twist.linear.x = self.max_linear_vel * min(1.0, wall_dist / self.wall_distance)
                    twist.angular.z = -self.kp_distance * error_distance
                    twist.angular.z = max(-self.max_angular_vel, 
                                         min(self.max_angular_vel, twist.angular.z))
                else:
                    twist.linear.x = 0.1
                    twist.angular.z = 0.3
        
        # Publicar comando
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()