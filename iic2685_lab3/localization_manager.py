#!/usr/bin/env python3
"""
Coordinador Principal de Localización
Maneja el proceso discreto de localización como se especifica en el laboratorio:
1. Ejecutar filtro por algunas iteraciones
2. Mover robot pequeña distancia/ángulo
3. Repetir hasta convergencia
"""
import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool, String
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


class LocalizationManager(Node):
    def __init__(self):
        super().__init__('localization_manager')
        
        # Parámetros del proceso de localización
        self.declare_parameter('move_distance', 0.3)      # Distancia de movimiento [m]
        self.declare_parameter('move_angle', 0.3)         # Ángulo de movimiento [rad]
        self.declare_parameter('iterations_per_move', 5)  # Iteraciones de filtro por movimiento
        
        self.move_distance = self.get_parameter('move_distance').get_parameter_value().double_value
        self.move_angle = self.get_parameter('move_angle').get_parameter_value().double_value
        self.iterations_per_move = self.get_parameter('iterations_per_move').get_parameter_value().integer_value
        
        # Estados del proceso
        self.INIT = 0
        self.FILTERING = 1
        self.MOVING = 2
        self.CONVERGED = 3
        
        self.current_state = self.INIT
        self.localization_active = True
        
        # Variables de control
        self.filter_iterations = 0
        self.total_moves = 0
        self.move_start_time = None
        self.move_duration = 0.0
        self.target_movement = None
        self.start_pose = None
        self.current_odom = None
        
        # Variables para el proceso discreto
        self.movement_sequence = [
            ('forward', self.move_distance),
            ('turn_left', self.move_angle),
            ('forward', self.move_distance),
            ('turn_right', self.move_angle * 2),
            ('forward', self.move_distance),
            ('turn_left', self.move_angle),
        ]
        self.sequence_index = 0
        
        # Suscriptores
        self.convergence_sub = self.create_subscription(
            Bool, '/localization_converged', self.convergence_callback, 10)
        self.estimated_pose_sub = self.create_subscription(
            PoseStamped, '/estimated_pose', self.estimated_pose_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel_mux/input/navigation', 10)
        self.status_pub = self.create_publisher(
            String, '/localization_status', 10)
        
        # Timer principal del proceso
        self.process_timer = self.create_timer(0.5, self.localization_process_loop)
        
        # Timer para el filtrado (simula las iteraciones del filtro)
        self.filter_timer = self.create_timer(1.0, self.filter_iteration_timer)
        
        self.get_logger().info("Coordinador de localización iniciado")
        self.get_logger().info("Proceso de localización discreto comenzando...")
    
    def convergence_callback(self, msg):
        """Callback cuando el filtro converge"""
        if msg.data and self.current_state != self.CONVERGED:
            self.current_state = self.CONVERGED
            self.localization_active = False
            self.get_logger().info("¡LOCALIZACIÓN COMPLETADA!")
            
            # Detener robot
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            
            # Publicar estado final
            status_msg = String()
            status_msg.data = "CONVERGED"
            self.status_pub.publish(status_msg)
    
    def estimated_pose_callback(self, msg):
        """Recibe la pose estimada del filtro"""
        if self.current_state == self.CONVERGED:
            # Imprimir pose final en coordenadas métricas
            x = msg.pose.position.x
            y = msg.pose.position.y
            
            orientation = msg.pose.orientation
            _, _, theta = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w])
            
            self.get_logger().info(f"POSE LOCALIZADA: x={x:.3f}m, y={y:.3f}m, θ={theta:.3f}rad")
    
    def odom_callback(self, msg):
        """Actualiza la odometría actual"""
        self.current_odom = msg
    
    def filter_iteration_timer(self):
        """Simula las iteraciones del filtro de partículas"""
        if self.current_state == self.FILTERING:
            self.filter_iterations += 1
            
            self.get_logger().info(f"Filtro de partículas - Iteración {self.filter_iterations}/{self.iterations_per_move}")
            
            if self.filter_iterations >= self.iterations_per_move:
                # Completar fase de filtrado
                self.filter_iterations = 0
                
                if not self.localization_active:
                    self.current_state = self.CONVERGED
                else:
                    self.current_state = self.MOVING
                    self.prepare_next_movement()
    
    def prepare_next_movement(self):
        """Prepara el siguiente movimiento en la secuencia"""
        if self.sequence_index >= len(self.movement_sequence):
            # Reiniciar secuencia
            self.sequence_index = 0
        
        movement_type, movement_value = self.movement_sequence[self.sequence_index]
        self.target_movement = (movement_type, movement_value)
        self.sequence_index += 1
        
        # Guardar pose inicial para el movimiento
        if self.current_odom:
            self.start_pose = self.current_odom.pose.pose
        
        self.move_start_time = time.time()
        
        # Calcular duración estimada del movimiento
        if movement_type == 'forward':
            # Tiempo basado en velocidad lineal estimada
            estimated_speed = 0.2  # m/s
            self.move_duration = movement_value / estimated_speed
        else:  # turn_left o turn_right
            # Tiempo basado en velocidad angular estimada
            estimated_angular_speed = 0.5  # rad/s
            self.move_duration = movement_value / estimated_angular_speed
        
        self.get_logger().info(f"Iniciando movimiento {self.total_moves + 1}: {movement_type} {movement_value:.2f}")
    
    def execute_movement(self):
        """Ejecuta el movimiento actual"""
        if not self.target_movement:
            return False
        
        current_time = time.time()
        movement_type, movement_value = self.target_movement
        
        cmd_vel = Twist()
        
        # Verificar si el movimiento debe continuar
        if current_time - self.move_start_time < self.move_duration:
            if movement_type == 'forward':
                cmd_vel.linear.x = 0.2  # Velocidad lineal moderada
                cmd_vel.angular.z = 0.0
            elif movement_type == 'turn_left':
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.5  # Velocidad angular moderada
            elif movement_type == 'turn_right':
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = -0.5
            
            self.cmd_vel_pub.publish(cmd_vel)
            return True  # Movimiento en progreso
        else:
            # Detener movimiento
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_vel)
            
            self.total_moves += 1
            self.get_logger().info(f"Movimiento completado. Total de movimientos: {self.total_moves}")
            
            # Resetear variables de movimiento
            self.target_movement = None
            self.move_start_time = None
            
            return False  # Movimiento completado
    
    def localization_process_loop(self):
        """Bucle principal del proceso de localización discreto"""
        if not self.localization_active:
            return
        
        # Máquina de estados del proceso
        if self.current_state == self.INIT:
            # Inicializar proceso
            self.get_logger().info("Iniciando fase de filtrado inicial...")
            self.current_state = self.FILTERING
            self.filter_iterations = 0
            
            # Publicar estado
            status_msg = String()
            status_msg.data = f"FILTERING_INITIAL"
            self.status_pub.publish(status_msg)
        
        elif self.current_state == self.FILTERING:
            # En proceso de filtrado - el timer de filtro maneja las iteraciones
            status_msg = String()
            status_msg.data = f"FILTERING_{self.filter_iterations}"
            self.status_pub.publish(status_msg)
        
        elif self.current_state == self.MOVING:
            # Ejecutar movimiento
            still_moving = self.execute_movement()
            
            if not still_moving:
                # Movimiento completado, volver a filtrar
                self.current_state = self.FILTERING
                self.filter_iterations = 0
                self.get_logger().info("Movimiento completado. Iniciando nueva fase de filtrado...")
            
            # Publicar estado
            status_msg = String()
            if still_moving and self.target_movement:
                movement_type = self.target_movement[0]
                status_msg.data = f"MOVING_{movement_type.upper()}"
            else:
                status_msg.data = "MOVING_COMPLETED"
            self.status_pub.publish(status_msg)
        
        elif self.current_state == self.CONVERGED:
            # Proceso completado
            status_msg = String()
            status_msg.data = "CONVERGED"
            self.status_pub.publish(status_msg)
    
    def get_movement_progress(self):
        """Calcula el progreso del movimiento actual"""
        if not self.move_start_time or not self.target_movement:
            return 0.0
        
        current_time = time.time()
        elapsed = current_time - self.move_start_time
        progress = min(elapsed / self.move_duration, 1.0)
        
        return progress


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LocalizationManager()
        
        # Mensaje inicial
        node.get_logger().info("="*50)
        node.get_logger().info("LABORATORIO 3: LOCALIZACIÓN CON FILTRO DE PARTÍCULAS")
        node.get_logger().info("="*50)
        node.get_logger().info("Proceso discreto de localización iniciado:")
        node.get_logger().info("1. Filtrar por algunas iteraciones")
        node.get_logger().info("2. Mover robot pequeña distancia/ángulo") 
        node.get_logger().info("3. Repetir hasta convergencia")
        node.get_logger().info("="*50)
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("Proceso de localización interrumpido por el usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()