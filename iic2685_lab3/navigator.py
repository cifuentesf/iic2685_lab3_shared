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
        
        # Parámetro
        self.linear_speed = 0.15  # Velocidad lineal base
        self.angular_speed = 1  # Velocidad angular máxima
        self.wall_distance = 0.35  
        self.min_front_distance = 0.35  
        self.intervalo_confianza = 0.85 
        
        # Estado
        self.exploring = True
        self.current_scan = None
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicador
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer - ejecutar navegación cada 100ms
        self.create_timer(0.1, self.navigate)
        
        self.get_logger().info("Navegador reactivo iniciado")
        
    def laser_callback(self, msg):
        self.current_scan = msg
        
    def confidence_callback(self, msg):
        if msg.data > self.intervalo_confianza and self.exploring:
            self.exploring = False
            self.stop_robot()
            self.get_logger().info(f"Localización completada (confianza: {msg.data:.2f})")
            
    def navigate(self):
        """Control de navegación principal"""
        if not self.exploring or self.current_scan is None:
            return
            
        # Procesar escaneo láser
        ranges = np.array(self.current_scan.ranges)
        
        # Manejar valores inválidos correctamente
        # Los valores inválidos deben tratarse como "sin obstáculo" (valor grande)
        for i in range(len(ranges)):
            if (ranges[i] < self.current_scan.range_min or 
                ranges[i] > self.current_scan.range_max or 
                np.isnan(ranges[i]) or 
                np.isinf(ranges[i])):
                ranges[i] = self.current_scan.range_max
        
        # Dividir en regiones (adaptado para láser de 180 grados)
        n = len(ranges)
        
        # Definir índices para cada región
        right_end = n // 6
        front_right_end = n // 3
        front_left_start = 2 * n // 3
        left_start = 5 * n // 6
        
        # Calcular distancias mínimas en cada región
        regions = {
            'right': np.min(ranges[0:right_end]) if right_end > 0 else self.current_scan.range_max,
            'front_right': np.min(ranges[right_end:front_right_end]) if front_right_end > right_end else self.current_scan.range_max,
            'front': np.min(ranges[front_right_end:front_left_start]) if front_left_start > front_right_end else self.current_scan.range_max,
            'front_left': np.min(ranges[front_left_start:left_start]) if left_start > front_left_start else self.current_scan.range_max,
            'left': np.min(ranges[left_start:]) if left_start < n else self.current_scan.range_max
        }
        
        # Crear comando de velocidad
        cmd = Twist()
        
        # Verificar si hay obstáculo frontal
        if regions['front'] < self.min_front_distance:
            # Obstáculo adelante - solo girar
            cmd.linear.x = 0.0
            # Girar hacia el lado con más espacio
            if regions['left'] > regions['right']:
                cmd.angular.z = self.angular_speed  # Girar a la izquierda
            else:
                cmd.angular.z = -self.angular_speed  # Girar a la derecha
            
            self.get_logger().debug(f"Obstáculo frontal detectado. Girando.")
            
        else:
            # Camino libre adelante - navegar
            cmd.linear.x = self.linear_speed
            
            # Lógica de seguimiento de pared derecha
            if regions['right'] < self.current_scan.range_max * 0.9:  # Hay pared a la derecha
                # Control proporcional para mantener distancia
                error = regions['right'] - self.wall_distance
                kp = 2.5  # Ganancia proporcional
                
                # Calcular velocidad angular
                cmd.angular.z = -kp * error  # Negativo porque queremos acercarnos si está lejos
                
                # Limitar velocidad angular
                cmd.angular.z = np.clip(cmd.angular.z, -self.angular_speed, self.angular_speed)
                
                # Si está muy cerca de la pared, reducir velocidad lineal
                if regions['right'] < self.wall_distance * 0.7:
                    cmd.linear.x *= 0.5
                    
            else:
                # No hay pared a la derecha - girar suavemente a la derecha para buscarla
                cmd.angular.z = -self.angular_speed * 0.3
                cmd.linear.x *= 0.7  # Reducir velocidad mientras busca pared
                
        # Debug
        self.get_logger().debug(
            f"Distancias - F:{regions['front']:.2f} R:{regions['right']:.2f} L:{regions['left']:.2f} | "
            f"Vel - Lin:{cmd.linear.x:.2f} Ang:{cmd.angular.z:.2f}"
        )
        
        # Publicar comando
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
        """Detener el robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        

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