#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        # Parámetros para movimiento discreto
        self.step_distance = 0.15        # Distancia fija pequeña [m]
        self.step_angle = np.pi/6        # Ángulo fijo pequeño [rad] (30°)
        self.linear_speed = 0.1          # Velocidad lenta para control preciso
        self.angular_speed = 0.2         # Velocidad angular lenta
        
        # Estado del algoritmo discreto
        self.state = "filtering"         # Estados: "filtering", "moving", "localized", "post_localization"
        self.current_scan = None
        self.filter_iterations = 0
        self.max_filter_iterations = 30  # Número de iteraciones del filtro antes de moverse
        self.step_count = 0
        self.movement_type = "forward"   # "forward", "turn_right", "turn_left"
        
        # Control de movimiento
        self.movement_start_time = None
        self.movement_duration = 0.0
        self.moving = False
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicador
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer para control secuencial
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador discreto iniciado - Iniciando fase de filtrado")
        
    def laser_callback(self, msg):
        """Callback para datos LIDAR"""
        self.current_scan = msg
        
    def confidence_callback(self, msg):
        """Callback para confianza de localización"""
        if msg.data > 0.8 and self.state != "localized" and self.state != "post_localization":
            self.state = "localized"
            self.get_logger().info(f"¡ROBOT LOCALIZADO! Confianza: {msg.data:.3f}")
            self.get_logger().info(f"Proceso de localización completado en {self.step_count} pasos")
            self.get_logger().info("Iniciando exploración post-localización...")
            
            # Esperar un momento y luego cambiar a exploración continua
            self.create_timer(3.0, self.start_post_localization, one_shot=True)
            
    def start_post_localization(self):
        """Iniciar exploración continua después de localización"""
        self.state = "post_localization"
        self.get_logger().info("Iniciando exploración continua con navegación reactiva")
            
    def control_loop(self):
        """Loop principal de control discreto y continuo"""
        if self.state == "localized":
            self.stop_robot()  # Mantener quieto durante transición
            return
        elif self.state == "post_localization":
            self.continuous_navigation()  # Navegación reactiva continua
            return
            
        # Lógica discreta original para localización
        if self.state == "filtering":
            self.filtering_phase()
        elif self.state == "moving":
            self.moving_phase()
            
    def continuous_navigation(self):
        """Navegación reactiva continua post-localización"""
        if self.current_scan is None:
            return
            
        ranges = np.array(self.current_scan.ranges)
        ranges[ranges >= self.current_scan.range_max] = self.current_scan.range_max
        ranges[ranges <= self.current_scan.range_min] = self.current_scan.range_max
        ranges[~np.isfinite(ranges)] = self.current_scan.range_max
        
        # Dividir escaneo en regiones
        n = len(ranges)
        front_ranges = ranges[n//3:2*n//3]
        front_min = np.min(front_ranges)
        
        cmd = Twist()
        
        # Lógica simple: avanzar o girar 90° derecha
        if front_min < 0.35:  # Obstáculo detectado
            # Girar 90° a la derecha
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5  # Velocidad angular para 90°
            self.get_logger().info("Obstáculo detectado - Girando 90° derecha")
        else:
            # Avanzar linealmente
            cmd.linear.x = 0.2  # Velocidad moderada
            cmd.angular.z = 0.0
            
        self.cmd_pub.publish(cmd)
            
    def control_loop(self):
        """Loop principal de control discreto"""
        if self.state == "localized":
            self.stop_robot()
            return
            
        if self.state == "filtering":
            self.filtering_phase()
        elif self.state == "moving":
            self.moving_phase()
            
    def filtering_phase(self):
        """Fase de filtrado: ejecutar filtro por varias iteraciones sin moverse"""
        self.stop_robot()  # Asegurar que el robot esté quieto
        
        self.filter_iterations += 1
        
        if self.filter_iterations >= self.max_filter_iterations:
            # Terminar fase de filtrado, iniciar movimiento
            self.get_logger().info(f"Completadas {self.filter_iterations} iteraciones de filtro")
            self.get_logger().info(f"Iniciando paso de movimiento #{self.step_count + 1}")
            
            # Resetear contador y cambiar a fase de movimiento
            self.filter_iterations = 0
            self.state = "moving"
            self.determine_next_movement()
            self.start_movement()
            
    def determine_next_movement(self):
        """Determinar el próximo movimiento basado en el entorno"""
        if self.current_scan is None:
            self.movement_type = "forward"
            return
            
        ranges = np.array(self.current_scan.ranges)
        ranges[ranges >= self.current_scan.range_max] = self.current_scan.range_max
        ranges[ranges <= self.current_scan.range_min] = self.current_scan.range_max
        ranges[~np.isfinite(ranges)] = self.current_scan.range_max
        
        # Dividir escaneo en regiones
        n = len(ranges)
        front_ranges = ranges[n//3:2*n//3]
        front_min = np.min(front_ranges)
        
        # Lógica simple: si hay obstáculo adelante, girar; sino, avanzar
        if front_min < 0.4:  # Obstáculo detectado
            # Alternar entre giro derecha e izquierda
            if self.step_count % 2 == 0:
                self.movement_type = "turn_right"
            else:
                self.movement_type = "turn_left"
        else:
            self.movement_type = "forward"
            
        self.get_logger().info(f"Movimiento seleccionado: {self.movement_type}")
        
    def start_movement(self):
        """Iniciar el movimiento calculando duración"""
        self.movement_start_time = time.time()
        self.moving = True
        
        if self.movement_type == "forward":
            # Calcular tiempo para recorrer step_distance
            self.movement_duration = self.step_distance / self.linear_speed
        else:  # turn_right o turn_left
            # Calcular tiempo para girar step_angle
            self.movement_duration = self.step_angle / self.angular_speed
            
    def moving_phase(self):
        """Fase de movimiento: ejecutar movimiento fijo"""
        if not self.moving:
            return
            
        current_time = time.time()
        elapsed = current_time - self.movement_start_time
        
        if elapsed < self.movement_duration:
            # Continuar movimiento
            cmd = Twist()
            
            if self.movement_type == "forward":
                cmd.linear.x = self.linear_speed
                cmd.angular.z = 0.0
            elif self.movement_type == "turn_right":
                cmd.linear.x = 0.0
                cmd.angular.z = -self.angular_speed
            elif self.movement_type == "turn_left":
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_speed
                
            self.cmd_pub.publish(cmd)
        else:
            # Movimiento completado
            self.stop_robot()
            self.moving = False
            self.step_count += 1
            
            self.get_logger().info(f"Movimiento completado. Volviendo a fase de filtrado...")
            
            # Volver a fase de filtrado
            self.state = "filtering"
            
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