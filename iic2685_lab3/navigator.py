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
        
        # Parámetros internos
        self.linear_speed = 0.15  # Reducido para mejor control
        self.angular_speed = 0.6  # Reducido para giros más suaves
        self.wall_distance = 0.35  # Distancia deseada a la pared
        self.min_front_distance = 0.4  # Distancia mínima frontal
        self.intervalo_confianza = 0.8
        
        # Estado
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
        exploring = False #self.exploring
        if msg.data > self.intervalo_confianza and exploring: #Encuentra localizacion
            self.exploring = False
            self.stop_robot()
            self.get_logger().info(f"Localización completada (confianza: {msg.data:.2f})")
            
    def navigate(self):
        if not self.exploring or self.current_scan is None:
            return
            
        # Procesar escaneo láser
        ranges = np.array(self.current_scan.ranges)
        # Reemplazar valores inválidos con un valor seguro (no el máximo)
        mask = (ranges < self.current_scan.range_min) | (ranges > self.current_scan.range_max) | np.isnan(ranges) | np.isinf(ranges)
        ranges[mask] = self.wall_distance * 2  # Valor seguro pero no infinito
        
        # Dividir en regiones
        n = len(ranges)
        regions = {
            'right': np.min(ranges[:n//6]),
            'front_right': np.min(ranges[n//6:n//3]),
            'front': np.min(ranges[n//3:2*n//3]),
            'front_left': np.min(ranges[2*n//3:5*n//6]),
            'left': np.min(ranges[5*n//6:])
        }
        
        # Lógica de navegación
        cmd = Twist()
        
        # Debug info
        self.get_logger().debug(f"Regiones: front={regions['front']:.2f}, right={regions['right']:.2f}, left={regions['left']:.2f}")
        
        if regions['front'] < self.min_front_distance:
            # Obstáculo
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed if regions['left'] > regions['right'] else -self.angular_speed
            self.get_logger().debug("Obstáculo frontal - girando")
        else:
            # Seguir pared derecha
            if regions['right'] > self.wall_distance * 1.5:
                # No hay pared cercana - buscarla
                if regions['front_right'] < regions['right']:
                    # Hay pared adelante a la derecha
                    cmd.linear.x = self.linear_speed
                    cmd.angular.z = 0.0
                else:
                    # Girar ligeramente hacia la derecha para buscar pared
                    cmd.linear.x = self.linear_speed * 0.7
                    cmd.angular.z = -self.angular_speed * 0.3
            else:
                # Hay pared - mantener distancia
                error = regions['right'] - self.wall_distance
                cmd.linear.x = self.linear_speed
                
                # Control proporcional con límites
                kp = 3.0
                cmd.angular.z = np.clip(-error * kp, -self.angular_speed, self.angular_speed)
                
                # Si muy cerca de la pared, reducir velocidad lineal
                if regions['right'] < self.wall_distance * 0.7:
                    cmd.linear.x *= 0.5
                
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