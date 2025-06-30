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
        
        # Parámetros de navegación (internos)
        self.step_distance = 0.15
        self.linear_speed = 0.12
        self.confidence_threshold = 0.8
        
        # Parámetros de navegación reactiva
        self.desired_wall_distance = 0.45
        self.collision_distance = 0.3
        self.front_collision_distance = 0.35
        
        # Control PID para seguimiento de pared (basado en simple_navigator)
        self.kp = 1.2
        self.ki = 0.01
        self.kd = 0.08
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_time = None
        
        # Estado del sistema
        self.state = "filtering"  # "filtering", "exploration", "localized"
        self.current_scan = None
        self.localization_confidence = 0.0
        self.filter_iterations = 0
        self.max_filter_iterations = 25
        self.step_count = 0
        self.localized_announced = False
        
        # Variables de movimiento discreto
        self.movement_start_time = None
        self.movement_duration = 0.0
        self.is_moving = False
        
        # Datos de sensores procesados (5 sectores como simple_navigator)
        self.left_distance = float('inf')
        self.left_front_distance = float('inf')
        self.front_distance = float('inf')
        self.right_front_distance = float('inf')
        self.right_distance = float('inf')
        
        # Variables de compatibilidad con simple_navigator
        self.left_wall_distance = float('inf')
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicador
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer principal
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador iniciado - Fase de filtrado inicial")

    def laser_callback(self, msg):
        """Procesar datos del LIDAR (idéntico a simple_navigator)"""
        self.current_scan = msg
        self.process_scan_data(msg)

    def process_scan_data(self, scan):
        """Procesar datos del LIDAR para navegación con 5 sectores"""
        if scan is None:
            return
            
        ranges = np.array(scan.ranges)
        ranges[~np.isfinite(ranges)] = 4.0
        ranges[ranges <= 0.0] = 4.0
        ranges[ranges > 3.9] = 4.0
        
        n = len(ranges)
        if n == 0:
            return
            
        # Dividir en 5 sectores (mismo que simple_navigator)
        sector_size = n // 5
        
        # Calcular distancias por sector usando percentiles para robustez
        sectors = {
            'left': ranges[4*sector_size:],
            'left_front': ranges[3*sector_size:4*sector_size],
            'front': ranges[2*sector_size:3*sector_size],
            'right_front': ranges[sector_size:2*sector_size],
            'right': ranges[:sector_size]
        }
        
        # Asignar distancias usando percentil 25 para evitar outliers
        for name, sector_data in sectors.items():
            valid_data = sector_data[sector_data < 3.9]
            if len(valid_data) > 0:
                distance = np.percentile(valid_data, 25)
            else:
                distance = float('inf')
                
            if name == 'left':
                self.left_distance = distance
                self.left_wall_distance = distance  # Compatibilidad con simple_navigator
            elif name == 'left_front':
                self.left_front_distance = distance
            elif name == 'front':
                self.front_distance = distance
            elif name == 'right_front':
                self.right_front_distance = distance
            elif name == 'right':
                self.right_distance = distance

    def confidence_callback(self, msg):
        """Procesar confianza de localización"""
        self.localization_confidence = msg.data
        
        # Transición de estados
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

    def estimate_robot_pose(self):
        """Estimación simple de pose del robot (para logging)"""
        # En un caso real, esto vendría del filtro de partículas
        # Para este ejemplo, usamos una estimación básica
        return [1.0, 1.0, 0.0]  # Placeholder

    def control_loop(self):
        """Bucle principal de control"""
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
        """Comportamiento durante fase de filtrado inicial"""
        # Robot quieto durante filtrado inicial
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        
        self.filter_iterations += 1

    def exploration_behavior(self):
        """Navegación discreta para exploración (movimientos paso a paso)"""
        current_time = self.get_clock().now()
        
        if not self.is_moving:
            # Decidir próximo movimiento
            self.decide_next_movement()
            self.movement_start_time = current_time
            self.is_moving = True
            
        else:
            # Ejecutar movimiento actual
            elapsed = (current_time - self.movement_start_time).nanoseconds * 1e-9
            
            if elapsed < self.movement_duration:
                self.execute_current_movement()
            else:
                # Finalizar movimiento
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.is_moving = False
                self.step_count += 1
                
                self.get_logger().info(
                    f"Paso {self.step_count} completado (confianza: {self.localization_confidence:.3f})"
                )

    def decide_next_movement(self):
        """Decidir próximo movimiento basado en sensores"""
        # Detección de obstáculo frontal crítico
        if self.front_distance < self.front_collision_distance:
            # Girar hacia donde hay más espacio
            if self.left_distance > self.right_distance:
                self.movement_type = "turn_left"
                self.movement_duration = 1.5  # Giro de ~90°
            else:
                self.movement_type = "turn_right" 
                self.movement_duration = 1.5
        else:
            # Avanzar
            self.movement_type = "forward"
            self.movement_duration = self.step_distance / self.linear_speed

    def execute_current_movement(self):
        """Ejecutar movimiento actual"""
        cmd = Twist()
        
        if self.movement_type == "forward":
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif self.movement_type == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6  # Velocidad angular para giro
        elif self.movement_type == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = -0.6
            
        self.cmd_pub.publish(cmd)

    def continuous_navigation(self):
        """Navegación reactiva continua (post-localización, basada en simple_navigator)"""
        if self.current_scan is None:
            return
            
        cmd = Twist()
        
        # Detección de obstáculo frontal crítico
        front_blocked = self.front_distance < self.front_collision_distance
        
        if front_blocked:
            # Maniobra evasiva
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            
            if left_space > right_space + 0.1:
                cmd.angular.z = 0.5   # Girar izquierda
                cmd.linear.x = 0.05   # Avance lento
            elif right_space > left_space + 0.1:
                cmd.angular.z = -0.5  # Girar derecha  
                cmd.linear.x = 0.05   # Avance lento
            else:
                cmd.angular.z = 0.6   # Giro agresivo izquierda (preferencia)
                cmd.linear.x = 0.0
                
        elif self.left_distance < 1.2:  # Hay pared izquierda para seguir
            # Seguimiento de pared izquierda con control PID (de simple_navigator)
            current_time = self.get_clock().now()
            
            if self.last_time is not None:
                dt = (current_time - self.last_time).nanoseconds * 1e-9
                
                if dt > 0:
                    # Error basado en distancia deseada a la pared
                    wall_error = self.left_distance - self.desired_wall_distance
                    
                    # Ajuste adicional si left_front está muy cerca
                    if self.left_front_distance < self.desired_wall_distance * 0.8:
                        wall_error += (self.desired_wall_distance * 0.8 - self.left_front_distance) * 0.5
                    
                    # Control PID
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
            # No hay pared cercana, buscar pared izquierda
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.3  # Giro suave a izquierda
        
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Detener robot completamente"""
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