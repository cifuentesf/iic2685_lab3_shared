#!/usr/bin/env python3
"""
Navegación Exploratoria Reactiva
Implementa navegación reactiva para explorar el entorno y ayudar al filtro de partículas
"""
import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import time


class ExplorationNavigator(Node):
    def __init__(self):
        super().__init__('exploration_navigator')
        
        # Parámetros de navegación
        self.declare_parameter('wall_distance', 0.5)  # Distancia objetivo a la pared [m]
        self.declare_parameter('linear_speed', 0.2)   # Velocidad lineal [m/s]
        self.declare_parameter('angular_speed', 0.5)  # Velocidad angular [rad/s]
        
        self.wall_distance = self.get_parameter('wall_distance').get_parameter_value().double_value
        self.linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        self.angular_speed = self.get_parameter('angular_speed').get_parameter_value().double_value
        
        # Estados de navegación
        self.FOLLOWING_WALL = 0
        self.TURNING = 1
        self.EXPLORING = 2
        self.STOPPED = 3
        
        self.current_state = self.EXPLORING
        self.navigation_enabled = True
        
        # Variables del sensor láser
        self.current_scan = None
        self.scan_ranges = []
        self.scan_angles = []
        
        # Variables para control
        self.turn_start_time = None
        self.turn_duration = 0.0
        self.obstacle_detected = False
        self.wall_following_side = "right"  # "left" o "right"
        
        # Suscriptores
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.convergence_sub = self.create_subscription(
            Bool, '/localization_converged', self.convergence_callback, 10)
        
        # Publicador
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel_mux/input/navigation', 10)
        
        # Timer para control de navegación
        self.control_timer = self.create_timer(0.1, self.navigation_control_loop)
        
        self.get_logger().info("Navegador exploratorio iniciado")
    
    def scan_callback(self, msg):
        """Procesa las lecturas del láser"""
        self.current_scan = msg
        self.scan_ranges = list(msg.ranges)
        
        # Calcular ángulos
        self.scan_angles = []
        for i in range(len(msg.ranges)):
            angle = msg.angle_min + i * msg.angle_increment
            self.scan_angles.append(angle)
    
    def convergence_callback(self, msg):
        """Detiene la navegación cuando el filtro converge"""
        if msg.data:
            self.navigation_enabled = False
            self.current_state = self.STOPPED
            self.get_logger().info("Filtro convergido - Deteniendo navegación")
    
    def get_laser_reading_at_angle(self, target_angle):
        """
        Obtiene la lectura del láser más cercana a un ángulo específico
        Args:
            target_angle: ángulo en radianes (-π/2 a π/2)
        Returns:
            distancia medida o None si no hay datos
        """
        if not self.scan_ranges or not self.scan_angles:
            return None
        
        # Encontrar el índice más cercano al ángulo objetivo
        angle_diffs = [abs(angle - target_angle) for angle in self.scan_angles]
        min_index = angle_diffs.index(min(angle_diffs))
        
        range_val = self.scan_ranges[min_index]
        
        # Verificar si la lectura es válida
        if (self.current_scan.range_min <= range_val <= self.current_scan.range_max):
            return range_val
        return None
    
    def get_front_distance(self):
        """Obtiene la distancia frontal mínima"""
        if not self.scan_ranges:
            return float('inf')
        
        # Considerar rayos frontales (aproximadamente ±30 grados)
        front_ranges = []
        for i, angle in enumerate(self.scan_angles):
            if abs(angle) <= math.pi/6:  # ±30 grados
                range_val = self.scan_ranges[i]
                if (self.current_scan.range_min <= range_val <= self.current_scan.range_max):
                    front_ranges.append(range_val)
        
        return min(front_ranges) if front_ranges else float('inf')
    
    def get_side_distance(self, side="right"):
        """
        Obtiene la distancia lateral
        Args:
            side: "right" o "left"
        """
        if side == "right":
            target_angle = -math.pi/2  # -90 grados
        else:
            target_angle = math.pi/2   # +90 grados
        
        return self.get_laser_reading_at_angle(target_angle)
    
    def detect_obstacle_ahead(self, min_distance=0.6):
        """Detecta si hay un obstáculo adelante"""
        front_dist = self.get_front_distance()
        return front_dist < min_distance
    
    def wall_following_control(self):
        """Implementa control de seguimiento de pared"""
        side_distance = self.get_side_distance(self.wall_following_side)
        
        if side_distance is None:
            # No hay lectura lateral válida, explorar
            self.current_state = self.EXPLORING
            return Twist()
        
        # Control PD simple para mantener distancia a la pared
        distance_error = side_distance - self.wall_distance
        
        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear_speed
        
        # Control angular proporcional al error de distancia
        kp = 1.0  # Ganancia proporcional
        cmd_vel.angular.z = -kp * distance_error
        
        # Limitar velocidad angular
        max_angular = self.angular_speed
        cmd_vel.angular.z = max(-max_angular, min(max_angular, cmd_vel.angular.z))
        
        return cmd_vel
    
    def exploration_control(self):
        """Control básico de exploración"""
        cmd_vel = Twist()
        
        # Buscar una dirección libre
        free_directions = []
        
        # Evaluar múltiples direcciones
        test_angles = np.linspace(-math.pi/3, math.pi/3, 7)  # -60° a +60°
        
        for angle in test_angles:
            distance = self.get_laser_reading_at_angle(angle)
            if distance and distance > 1.0:  # Distancia mínima segura
                free_directions.append((angle, distance))
        
        if free_directions:
            # Elegir la dirección con mayor distancia libre
            best_direction = max(free_directions, key=lambda x: x[1])
            target_angle = best_direction[0]
            
            cmd_vel.linear.x = self.linear_speed * 0.7  # Velocidad reducida en exploración
            cmd_vel.angular.z = 2.0 * target_angle  # Girar hacia la dirección libre
            
            # Verificar si hay una pared cerca para seguir
            right_dist = self.get_side_distance("right")
            left_dist = self.get_side_distance("left")
            
            if right_dist and right_dist < 1.5:
                self.wall_following_side = "right"
                self.current_state = self.FOLLOWING_WALL
            elif left_dist and left_dist < 1.5:
                self.wall_following_side = "left"
                self.current_state = self.FOLLOWING_WALL
        else:
            # No hay direcciones libres, girar in situ
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed
        
        return cmd_vel
    
    def turning_control(self):
        """Control durante giros de 90 grados"""
        current_time = time.time()
        
        if self.turn_start_time is None:
            self.turn_start_time = current_time
            self.turn_duration = math.pi/2 / self.angular_speed  # Tiempo para 90°
        
        cmd_vel = Twist()
        
        if current_time - self.turn_start_time < self.turn_duration:
            # Continuar girando
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed
        else:
            # Terminar giro
            self.turn_start_time = None
            self.current_state = self.EXPLORING
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        
        return cmd_vel
    
    def navigation_control_loop(self):
        """Bucle principal de control de navegación"""
        if not self.navigation_enabled or self.current_scan is None:
            # Detener robot
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return
        
        # Detectar obstáculos frontales
        obstacle_ahead = self.detect_obstacle_ahead()
        
        # Máquina de estados para navegación
        if self.current_state == self.STOPPED:
            cmd_vel = Twist()  # Robot detenido
        
        elif self.current_state == self.FOLLOWING_WALL:
            if obstacle_ahead:
                # Obstáculo adelante, comenzar giro
                self.current_state = self.TURNING
                cmd_vel = self.turning_control()
            else:
                cmd_vel = self.wall_following_control()
        
        elif self.current_state == self.EXPLORING:
            if obstacle_ahead:
                # Obstáculo adelante, comenzar giro
                self.current_state = self.TURNING
                cmd_vel = self.turning_control()
            else:
                cmd_vel = self.exploration_control()
        
        elif self.current_state == self.TURNING:
            cmd_vel = self.turning_control()
        
        else:
            cmd_vel = Twist()
        
        # Publicar comando de velocidad
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Debug info (ocasional)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 50 == 0:  # Cada 5 segundos
            state_names = {
                self.FOLLOWING_WALL: "SIGUIENDO_PARED",
                self.TURNING: "GIRANDO", 
                self.EXPLORING: "EXPLORANDO",
                self.STOPPED: "DETENIDO"
            }
            
            front_dist = self.get_front_distance()
            side_dist = self.get_side_distance(self.wall_following_side)
            
            self.get_logger().info(
                f"Estado: {state_names.get(self.current_state, 'DESCONOCIDO')}, "
                f"Dist. frontal: {front_dist:.2f}m, "
                f"Dist. lateral: {side_dist:.2f}m" if side_dist else f"Dist. frontal: {front_dist:.2f}m"
            )


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ExplorationNavigator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()