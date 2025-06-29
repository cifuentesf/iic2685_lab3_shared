#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class ReactiveNavigator(Node):
    
    def __init__(self):
        super().__init__('reactive_navigator')
        
        # Parámetros
        self.linear_speed = 0.2
        self.angular_speed = 0.8
        self.wall_distance = 0.4
        self.min_front_distance = 0.5
        self.confidence_threshold = 0.8
        self.exploring = True
        self.current_scan = None
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicador
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_mux/input/navigation', 10)
        
        # Timer
        self.create_timer(0.1, self.navigate)
        
        self.get_logger().info("Navegador reactivo iniciado")
        
    def laser_callback(self, msg):
        self.current_scan = msg
        
    def confidence_callback(self, msg):
        """Se encarga de detener el bot y printear localización"""
        if msg.data > self.confidence_threshold and self.exploring:
            self.exploring = False
            self.stop_robot()
            self.get_logger().info(f"Localización completada (confianza: {msg.data:.2f})")
            
    def navigate(self):
        """Control de navegación principal"""
        if not self.exploring or self.current_scan is None:
            return
            
        ranges = np.array(self.current_scan.ranges)
        ranges[ranges > self.current_scan.range_max] = self.current_scan.range_max
        ranges[ranges < self.current_scan.range_min] = self.current_scan.range_max
        
        n = len(ranges)
        regions = {
            'right': np.min(ranges[:n//6]),
            'front_right': np.min(ranges[n//6:n//3]),
            'front': np.min(ranges[n//3:2*n//3]),
            'front_left': np.min(ranges[2*n//3:5*n//6]),
            'left': np.min(ranges[5*n//6:])
        }
        

        cmd = Twist()
        
        if regions['front'] < self.min_front_distance:
            # Obstáculo adelante - girar
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed if regions['left'] > regions['right'] else -self.angular_speed
        else:
            # Seguir pared derecha
            if regions['right'] > 2 * self.wall_distance:
                # No hay pared - girar hacia ella
                cmd.linear.x = self.linear_speed * 0.5
                cmd.angular.z = -self.angular_speed * 0.5
            else:
                # Mantener distancia a la pared
                error = regions['right'] - self.wall_distance
                cmd.linear.x = self.linear_speed
                cmd.angular.z = np.clip(-error * 2.0, -self.angular_speed, self.angular_speed)
                
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
        """Detener robot"""
        self.cmd_vel_pub.publish(Twist())
        

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()