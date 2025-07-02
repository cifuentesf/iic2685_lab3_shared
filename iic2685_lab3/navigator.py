#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        # Parámetros de navegación
        self.step_distance = 0.15
        self.linear_speed = 0.12
        self.confidence_threshold = 0.8
        self.desired_wall_distance = 0.45
        self.collision_distance = 0.3
        self.front_collision_distance = 0.35
        
        # Parámetros PID para seguimiento de pared
        self.kp = 1.2
        self.ki = 0.01
        self.kd = 0.08
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_time = None
        
        # Estado del navegador
        self.state = "filtering"  # "filtering", "exploration", "localized"
        self.current_scan = None
        self.localization_confidence = 0.0
        self.filter_iterations = 0
        self.max_filter_iterations = 25
        self.step_count = 0
        self.localized_announced = False
        
        # Variables de movimiento
        self.movement_start_time = None
        self.movement_duration = 0.0
        self.is_moving = False
        
        # Distancias del láser
        self.left_distance = float('inf')
        self.left_front_distance = float('inf')
        self.front_distance = float('inf')
        self.right_front_distance = float('inf')
        self.right_distance = float('inf')
        self.left_wall_distance = float('inf')
        
        # Variables para mejor pose - CORRECCIÓN
        self.best_pose = None
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.confidence_sub = self.create_subscription(
            Float64, '/localization_confidence', self.confidence_callback, 10)
        self.best_pose_sub = self.create_subscription(
            PointStamped, '/best_pose', self.best_pose_callback, 10)  # CORRECCIÓN: PointStamped
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer para control
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador iniciado")

    def laser_callback(self, msg):
        """Callback para datos del láser"""
        self.current_scan = msg
        self.process_scan_data(msg)

    def process_scan_data(self, scan):
        """Procesar datos del láser para extraer distancias por sectores"""
        if scan is None:
            return
        
        ranges = np.array(scan.ranges)
        ranges[~np.isfinite(ranges)] = 4.0
        ranges[ranges <= 0.0] = 4.0
        ranges[ranges > 3.9] = 4.0
        
        n = len(ranges)
        if n == 0:
            return
        
        # Dividir en 5 sectores
        sector_size = n // 5
        
        sectors = {
            'left': ranges[4*sector_size:],
            'left_front': ranges[3*sector_size:4*sector_size],
            'front': ranges[2*sector_size:3*sector_size],
            'right_front': ranges[sector_size:2*sector_size],
            'right': ranges[:sector_size]
        }
        
        # Calcular distancias mínimas por sector
        self.left_distance = np.min(sectors['left']) if len(sectors['left']) > 0 else float('inf')
        self.left_front_distance = np.min(sectors['left_front']) if len(sectors['left_front']) > 0 else float('inf')
        self.front_distance = np.min(sectors['front']) if len(sectors['front']) > 0 else float('inf')
        self.right_front_distance = np.min(sectors['right_front']) if len(sectors['right_front']) > 0 else float('inf')
        self.right_distance = np.min(sectors['right']) if len(sectors['right']) > 0 else float('inf')
        
        # Distancia promedio a la pared izquierda para seguimiento
        self.left_wall_distance = np.mean(sectors['left']) if len(sectors['left']) > 0 else float('inf')

    def confidence_callback(self, msg):
        """Callback para confianza de localización"""
        self.localization_confidence = msg.data
        
        # Transiciones de estado
        if self.state == "filtering" and self.localization_confidence > self.confidence_threshold:
            if not self.localized_announced:
                best_pose = self.estimate_robot_pose()
                self.get_logger().info(
                    f"¡Robot localizado! Confianza: {self.localization_confidence:.3f}, "
                    f"Pose estimada: x={best_pose[0]:.3f}, y={best_pose[1]:.3f}, θ={best_pose[2]:.3f}"
                )
                self.localized_announced = True
                self.state = "localized"
        elif self.state == "filtering" and self.filter_iterations >= self.max_filter_iterations:
            self.state = "exploration"
            self.get_logger().info("Iniciando exploración reactiva")

    def best_pose_callback(self, msg):
        """Callback para mejor pose estimada - CORRECCIÓN"""
        self.best_pose = msg

    def estimate_robot_pose(self):
        """Estimar pose del robot - CORRECCIÓN"""
        if self.best_pose is not None:
            x = self.best_pose.point.x  # CORRECCIÓN: point.x en lugar de linear.x
            y = self.best_pose.point.y  # CORRECCIÓN: point.y en lugar de linear.y
            theta = 0.0  # PointStamped no incluye orientación
        else:
            x = y = theta = 0.0
        return (x, y, theta)

    def control_loop(self):
        """Bucle principal de control"""
        if self.current_scan is None:
            return
        
        if self.state == "filtering":
            self.filtering_behavior()
        elif self.state == "exploration":
            self.exploration_behavior()
        elif self.state == "localized":
            self.continuous_navigation()

    def filtering_behavior(self):
        """Comportamiento durante la fase de filtrado (robot estático)"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.filter_iterations += 1

    def exploration_behavior(self):
        """Comportamiento de exploración reactiva"""
        current_time = self.get_clock().now()
        
        if not self.is_moving:
            self.decide_next_movement()
            self.movement_start_time = current_time
            self.is_moving = True
        else:
            elapsed = (current_time - self.movement_start_time).nanoseconds * 1e-9
            if elapsed < self.movement_duration:
                self.execute_current_movement()
            else:
                # Movimiento completado
                self.is_moving = False
                self.step_count += 1

    def decide_next_movement(self):
        """Decidir próximo movimiento basado en sensores"""
        # Movimiento reactivo simple
        if self.front_distance < self.front_collision_distance:
            # Hay obstáculo al frente, rotar
            if self.left_distance > self.right_distance:
                self.current_movement = "turn_left"
                self.movement_duration = 1.0  # 1 segundo de rotación
            else:
                self.current_movement = "turn_right"
                self.movement_duration = 1.0
        else:
            # Camino despejado, avanzar
            self.current_movement = "forward"
            self.movement_duration = self.step_distance / self.linear_speed

    def execute_current_movement(self):
        """Ejecutar movimiento actual"""
        cmd = Twist()
        
        if self.current_movement == "forward":
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif self.current_movement == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # rad/s
        elif self.current_movement == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5  # rad/s
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        self.cmd_pub.publish(cmd)

    def continuous_navigation(self):
        """Navegación continua una vez localizado (seguimiento de pared)"""
        cmd = Twist()
        
        # Verificar colisiones
        if (self.front_distance < self.collision_distance or 
            self.left_front_distance < self.collision_distance or 
            self.right_front_distance < self.collision_distance):
            
            # Parar y rotar para evitar colisión
            cmd.linear.x = 0.0
            if self.left_distance > self.right_distance:
                cmd.angular.z = 0.8
            else:
                cmd.angular.z = -0.8
        else:
            # Seguimiento de pared izquierda con control PID
            error = self.desired_wall_distance - self.left_wall_distance
            
            # Control PID simple
            self.integral_error += error * 0.1  # dt = 0.1s
            derivative_error = error - self.previous_error
            
            angular_velocity = (self.kp * error + 
                              self.ki * self.integral_error + 
                              self.kd * derivative_error)
            
            # Limitar velocidad angular
            angular_velocity = max(-1.0, min(1.0, angular_velocity))
            
            cmd.linear.x = self.linear_speed
            cmd.angular.z = angular_velocity
            
            self.previous_error = error
        
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