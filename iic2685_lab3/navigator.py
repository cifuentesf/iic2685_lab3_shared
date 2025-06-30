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
        
        # Parámetros de navegación (internos, no externos)
        self.step_distance = 0.15
        self.linear_speed = 0.12
        self.angular_speed = 0.3
        self.confidence_threshold = 0.8
        
        # Parámetros de navegación reactiva (menos restrictivos)
        self.desired_wall_distance = 0.4
        self.collision_distance = 0.25  # Reducido de 0.35 para ser menos restrictivo
        self.front_collision_distance = 0.3  # Específico para sector frontal
        self.wall_following_active = True
        
        # Control PID simplificado para seguimiento de pared
        self.kp = 1.2
        self.ki = 0.01
        self.kd = 0.08
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_time = None
        
        # Estado del sistema
        self.state = "filtering"  # "filtering", "moving", "localized"
        self.current_scan = None
        self.localization_confidence = 0.0
        self.filter_iterations = 0
        self.max_filter_iterations = 25
        self.step_count = 0
        
        # Variables de movimiento discreto
        self.movement_start_time = None
        self.movement_duration = 0.0
        self.movement_type = "forward"
        self.is_moving = False
        
        # Datos de sensores procesados (5 sectores)
        self.left_distance = float('inf')          # Sector izquierdo (144-180°)
        self.left_front_distance = float('inf')    # Sector izquierdo-frontal (108-144°)
        self.front_distance = float('inf')         # Sector frontal (72-108°)
        self.right_front_distance = float('inf')   # Sector derecho-frontal (36-72°)
        self.right_distance = float('inf')         # Sector derecho (0-36°)
        
        # Variables de compatibilidad
        self.left_wall_distance = float('inf')
        self.right_wall_distance = float('inf')
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicador
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer principal
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador mejorado iniciado - Fase de filtrado inicial")

    def laser_callback(self, msg):
        """Procesar datos del LIDAR y extraer información útil"""
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
            
        # Dividir escaneo en 5 sectores
        sector_size = n // 5
        
        # Sector 0: Right (0-36°)
        right_sector = ranges[0:sector_size]
        self.right_distance = np.percentile(right_sector, 25) if len(right_sector) > 0 else float('inf')
        
        # Sector 1: Right-Front (36-72°)
        right_front_sector = ranges[sector_size:2*sector_size]
        self.right_front_distance = np.percentile(right_front_sector, 25) if len(right_front_sector) > 0 else float('inf')
        
        # Sector 2: Front (72-108°)
        front_sector = ranges[2*sector_size:3*sector_size]
        self.front_distance = np.percentile(front_sector, 15) if len(front_sector) > 0 else float('inf')
        
        # Sector 3: Left-Front (108-144°)
        left_front_sector = ranges[3*sector_size:4*sector_size]
        self.left_front_distance = np.percentile(left_front_sector, 25) if len(left_front_sector) > 0 else float('inf')
        
        # Sector 4: Left (144-180°)
        left_sector = ranges[4*sector_size:]
        self.left_distance = np.percentile(left_sector, 25) if len(left_sector) > 0 else float('inf')
        
        # Mantener compatibilidad con variables anteriores
        self.left_wall_distance = self.left_distance
        self.right_wall_distance = self.right_distance

    def confidence_callback(self, msg):
        """Callback para confianza de localización"""
        self.localization_confidence = msg.data
        
        if msg.data >= self.confidence_threshold and self.state != "localized":
            self.state = "localized"
            self.get_logger().info(f"¡Robot localizado! Confianza: {msg.data:.3f}")
            self.get_logger().info(f"Localización completada en {self.step_count} pasos")
            # Continuar con navegación reactiva después de localización
            self.create_timer(2.0, self.start_post_localization_navigation, one_shot=True)

    def start_post_localization_navigation(self):
        """Iniciar navegación continua después de localización"""
        self.state = "exploring"
        self.get_logger().info("Iniciando exploración continua post-localización")

    def control_loop(self):
        """Loop principal de control"""
        if self.state == "filtering":
            self.filtering_phase()
        elif self.state == "moving":
            self.moving_phase()
        elif self.state == "localized":
            self.stop_robot()  # Pausa breve antes de exploración
        elif self.state == "exploring":
            self.continuous_navigation()

    def filtering_phase(self):
        """Fase de filtrado: robot quieto mientras el filtro procesa"""
        self.stop_robot()
        self.filter_iterations += 1
        
        if self.filter_iterations >= self.max_filter_iterations:
            self.get_logger().info(f"Completadas {self.filter_iterations} iteraciones de filtro")
            self.get_logger().info(f"Iniciando movimiento #{self.step_count + 1}")
            
            self.filter_iterations = 0
            self.state = "moving"
            self.determine_movement()
            self.execute_movement()

    def determine_movement(self):
        """Determinar tipo de movimiento basado en 5 sectores del LIDAR (menos restrictivo)"""
        if self.current_scan is None:
            self.movement_type = "forward"
            return
            
        # Lógica mejorada con umbrales menos restrictivos
        # Detectar obstáculo frontal crítico (solo sector FRONT)
        front_critical = self.front_distance < self.front_collision_distance
        
        # Detectar obstáculos laterales frontales
        lateral_front_obstruction = (self.left_front_distance < self.collision_distance or 
                                   self.right_front_distance < self.collision_distance)
        
        if front_critical:
            # Obstáculo frontal crítico - DEBE girar
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            
            if left_space > right_space + 0.1:
                self.movement_type = "turn_left"
            elif right_space > left_space + 0.1:
                self.movement_type = "turn_right"
            else:
                # Espacios similares - preferir izquierda para seguimiento de pared
                self.movement_type = "turn_left"
                
        elif lateral_front_obstruction and self.front_distance < 0.5:
            # Obstáculos laterales-frontales Y frente relativamente cerca
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            
            if left_space > right_space + 0.15:
                self.movement_type = "turn_left"
            elif right_space > left_space + 0.15:
                self.movement_type = "turn_right"
            else:
                # Si no hay clara ventaja, intentar avanzar
                self.movement_type = "forward"
                
        else:
            # Camino relativamente libre - navegar basado en seguimiento de pared
            if self.left_distance < 0.8:  # Hay pared izquierda para seguir
                wall_error = self.left_distance - self.desired_wall_distance
                
                if abs(wall_error) < 0.12:  # Margen más amplio para avanzar
                    self.movement_type = "forward"
                elif wall_error > 0.2:  # Bastante lejos de la pared
                    self.movement_type = "turn_left"
                elif wall_error < -0.2:  # Muy cerca de la pared
                    self.movement_type = "turn_right"
                else:
                    self.movement_type = "forward"  # Caso por defecto: avanzar
            else:
                # No hay pared izquierda clara - decidir basado en espacios
                if self.front_distance > 0.8:  # Frente bastante libre
                    if self.right_distance < self.left_distance and self.right_distance < 1.0:
                        # Hay pared derecha - seguirla temporalmente
                        self.movement_type = "forward"
                    else:
                        # Buscar pared girando suavemente
                        self.movement_type = "turn_left"
                else:
                    # Frente no tan libre pero navegable
                    self.movement_type = "forward"
        
        # Log mejorado con análisis de decisión
        self.get_logger().info(
            f"Mov: {self.movement_type} | "
            f"L:{self.left_distance:.2f} LF:{self.left_front_distance:.2f} "
            f"F:{self.front_distance:.2f} RF:{self.right_front_distance:.2f} "
            f"R:{self.right_distance:.2f} | "
            f"FCrit:{front_critical} LFObs:{lateral_front_obstruction}"
        )

    def execute_movement(self):
        """Ejecutar movimiento discreto seleccionado"""
        self.movement_start_time = self.get_clock().now()
        self.is_moving = True
        
        # Calcular duración del movimiento
        if self.movement_type == "forward":
            self.movement_duration = self.step_distance / self.linear_speed
        else:  # Giros
            rotation_angle = np.pi/3  # 60 grados
            self.movement_duration = rotation_angle / self.angular_speed
        
        self.send_movement_command()

    def send_movement_command(self):
        """Enviar comando de velocidad según tipo de movimiento"""
        cmd = Twist()
        
        if self.movement_type == "forward":
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif self.movement_type == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        elif self.movement_type == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = -self.angular_speed
        
        self.cmd_pub.publish(cmd)

    def moving_phase(self):
        """Fase de movimiento: ejecutar movimiento por tiempo determinado"""
        if not self.is_moving:
            return
            
        current_time = self.get_clock().now()
        elapsed = (current_time - self.movement_start_time).nanoseconds * 1e-9
        
        if elapsed < self.movement_duration:
            # Continuar movimiento
            self.send_movement_command()
        else:
            # Terminar movimiento
            self.stop_robot()
            self.is_moving = False
            self.step_count += 1
            
            # Volver a fase de filtrado
            self.state = "filtering"
            self.get_logger().info(f"Movimiento completado. Volviendo a filtrado (paso {self.step_count})")

    def continuous_navigation(self):
        """Navegación reactiva continua con 5 sectores después de localización"""
        if self.current_scan is None:
            return
            
        cmd = Twist()
        
        # Detectar obstáculos frontales usando 3 sectores frontales
        frontal_clear = (self.front_distance > self.collision_distance and 
                        self.left_front_distance > self.collision_distance and 
                        self.right_front_distance > self.collision_distance)
        
        if not frontal_clear:
            # Obstáculo frontal detectado - maniobra evasiva
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            
            if left_space > right_space + 0.1:
                # Más espacio a la izquierda
                cmd.angular.z = 0.5
                cmd.linear.x = 0.05  # Avance muy lento mientras gira
            elif right_space > left_space + 0.1:
                # Más espacio a la derecha  
                cmd.angular.z = -0.5
                cmd.linear.x = 0.05
            else:
                # Espacios similares - giro más agresivo hacia la izquierda (preferencia)
                cmd.angular.z = 0.6
                cmd.linear.x = 0.0
                
        elif self.left_distance < 1.2:  # Hay pared izquierda para seguir
            # Seguimiento de pared izquierda con control PID mejorado
            current_time = self.get_clock().now()
            
            if self.last_time is not None:
                dt = (current_time - self.last_time).nanoseconds * 1e-9
                
                if dt > 0:
                    # Error basado en distancia deseada a la pared
                    wall_error = self.left_distance - self.desired_wall_distance
                    
                    # Ajuste adicional si el sector left_front está muy cerca
                    if self.left_front_distance < self.desired_wall_distance * 0.8:
                        wall_error -= 0.1  # Forzar alejamiento
                    
                    # Control PID
                    proportional = self.kp * wall_error
                    self.integral_error += wall_error * dt
                    self.integral_error = np.clip(self.integral_error, -0.5, 0.5)  # Anti-windup
                    integral = self.ki * self.integral_error
                    derivative = self.kd * (wall_error - self.previous_error) / dt
                    
                    # Comando angular (negativo porque queremos alejarnos cuando error es positivo)
                    angular_output = -(proportional + integral + derivative)
                    cmd.angular.z = np.clip(angular_output, -0.7, 0.7)
                    
                    # Ajustar velocidad lineal basada en qué tan bien estamos siguiendo
                    if abs(wall_error) < 0.1:
                        cmd.linear.x = self.linear_speed  # Velocidad normal
                    elif abs(wall_error) < 0.2:
                        cmd.linear.x = self.linear_speed * 0.8  # Velocidad reducida
                    else:
                        cmd.linear.x = self.linear_speed * 0.6  # Velocidad muy reducida
                    
                    self.previous_error = wall_error
            
            self.last_time = current_time
            
        else:
            # No hay pared izquierda - buscar pared o explorar
            if self.right_distance < self.left_distance and self.right_distance < 1.5:
                # Hay pared derecha más cerca - acercarse gradualmente
                cmd.linear.x = self.linear_speed * 0.8
                cmd.angular.z = -0.15  # Giro suave hacia la derecha
            else:
                # Explorar girando hacia la izquierda para buscar pared
                cmd.linear.x = self.linear_speed * 0.9
                cmd.angular.z = 0.2  # Giro suave hacia la izquierda
        
        # Publicar comando
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Detener completamente el robot"""
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