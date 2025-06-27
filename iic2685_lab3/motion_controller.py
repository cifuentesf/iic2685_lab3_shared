#!/usr/bin/env python3
"""
Actividad 3: Exploración Reactiva para Localización
Implementa navegación reactiva para explorar el entorno y mejorar la localización
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
import numpy as np
from enum import Enum
import time

class RobotState(Enum):
    """Estados del robot para la máquina de estados"""
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    WALL_FOLLOW_LEFT = 4
    WALL_FOLLOW_RIGHT = 5
    ROTATE_TO_FIND_PATH = 6

class ReactiveExplorer(Node):
    def __init__(self):
        super().__init__('reactive_explorer')
        self.get_logger().info("Explorador reactivo para localización iniciado")

        # Parámetros de control
        self.linear_speed = 0.15      # Velocidad lineal (m/s)
        self.angular_speed = 0.5      # Velocidad angular (rad/s)
        self.wall_distance = 0.4      # Distancia deseada a la pared (m)
        self.obstacle_threshold = 0.35 # Distancia mínima a obstáculos (m)
        self.wall_follow_kp = 2.0     # Ganancia proporcional para seguimiento de pared
        
        # Estado del robot
        self.state = RobotState.MOVE_FORWARD
        self.following_wall = "left"  # Qué pared seguir por defecto
        self.exploration_complete = False
        
        # Publicadores y suscriptores
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel_mux/input/navigation', 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.convergence_sub = self.create_subscription(
            Float32, '/particle_convergence', self.convergence_callback, 10
        )
        
        # Publicador de estado
        self.state_pub = self.create_publisher(
            Bool, '/exploration_active', 10
        )
        
        # Variables del sensor
        self.scan_data = None
        self.regions = {
            'right': float('inf'),
            'front_right': float('inf'),
            'front': float('inf'),
            'front_left': float('inf'),
            'left': float('inf')
        }
        
        # Control de convergencia
        self.convergence_value = float('inf')
        self.convergence_threshold = 0.3  # metros
        
        # Timer principal
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Variables para evitar quedar atrapado
        self.last_state_change = time.time()
        self.state_duration = 0.0
        self.max_state_duration = 10.0  # segundos máximos en un estado

    def lidar_callback(self, msg):
        """Procesa datos del LIDAR y extrae regiones de interés"""
        self.scan_data = msg
        
        # Dividir el scan en regiones
        ranges = np.array(msg.ranges)
        
        # Eliminar valores NaN e Inf
        ranges[np.isnan(ranges)] = msg.range_max
        ranges[np.isinf(ranges)] = msg.range_max
        
        # Definir regiones (asumiendo 180 grados de -90 a 90)
        total_points = len(ranges)
        regions_size = total_points // 5
        
        # Calcular mínimos por región
        self.regions = {
            'right': np.min(ranges[0:regions_size]),
            'front_right': np.min(ranges[regions_size:2*regions_size]),
            'front': np.min(ranges[2*regions_size:3*regions_size]),
            'front_left': np.min(ranges[3*regions_size:4*regions_size]),
            'left': np.min(ranges[4*regions_size:])
        }

    def convergence_callback(self, msg):
        """Recibe el valor de convergencia del filtro de partículas"""
        self.convergence_value = msg.data
        
        # Verificar si hemos convergido
        if self.convergence_value < self.convergence_threshold:
            if not self.exploration_complete:
                self.get_logger().info(
                    f"¡Localización exitosa! Convergencia: {self.convergence_value:.3f} m"
                )
                self.exploration_complete = True

    def determine_state(self):
        """Máquina de estados para determinar el comportamiento del robot"""
        if self.scan_data is None:
            return self.state
        
        # Si ya convergimos, detener exploración
        if self.exploration_complete:
            return RobotState.MOVE_FORWARD  # Estado idle
        
        # Verificar tiempo en el estado actual
        current_time = time.time()
        self.state_duration = current_time - self.last_state_change
        
        # Cambiar estado si llevamos mucho tiempo
        if self.state_duration > self.max_state_duration:
            self.get_logger().warn("Tiempo máximo en estado alcanzado, rotando...")
            self.last_state_change = current_time
            return RobotState.ROTATE_TO_FIND_PATH
        
        # Lógica de transición de estados
        front_blocked = self.regions['front'] < self.obstacle_threshold
        front_left_blocked = self.regions['front_left'] < self.obstacle_threshold
        front_right_blocked = self.regions['front_right'] < self.obstacle_threshold
        left_clear = self.regions['left'] > self.wall_distance
        right_clear = self.regions['right'] > self.wall_distance
        
        # Prioridad: seguir pared > evitar obstáculos > explorar
        
        if self.state == RobotState.MOVE_FORWARD:
            if front_blocked:
                # Decidir dirección de giro basado en espacio disponible
                if self.regions['left'] > self.regions['right']:
                    return RobotState.TURN_LEFT
                else:
                    return RobotState.TURN_RIGHT
            elif not left_clear:
                return RobotState.WALL_FOLLOW_LEFT
            elif not right_clear:
                return RobotState.WALL_FOLLOW_RIGHT
                
        elif self.state == RobotState.TURN_LEFT:
            if not front_blocked and not front_left_blocked:
                return RobotState.MOVE_FORWARD
                
        elif self.state == RobotState.TURN_RIGHT:
            if not front_blocked and not front_right_blocked:
                return RobotState.MOVE_FORWARD
                
        elif self.state == RobotState.WALL_FOLLOW_LEFT:
            if front_blocked:
                return RobotState.TURN_RIGHT
            elif left_clear:
                return RobotState.TURN_LEFT  # Mantener contacto con la pared
                
        elif self.state == RobotState.WALL_FOLLOW_RIGHT:
            if front_blocked:
                return RobotState.TURN_LEFT
            elif right_clear:
                return RobotState.TURN_RIGHT  # Mantener contacto con la pared
                
        elif self.state == RobotState.ROTATE_TO_FIND_PATH:
            if not front_blocked:
                return RobotState.MOVE_FORWARD
        
        return self.state

    def execute_state(self):
        """Ejecuta la acción correspondiente al estado actual"""
        cmd = Twist()
        
        if self.exploration_complete:
            # Detener el robot si ya convergimos
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            if self.state == RobotState.MOVE_FORWARD:
                cmd.linear.x = self.linear_speed
                cmd.angular.z = 0.0
                
            elif self.state == RobotState.TURN_LEFT:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_speed
                
            elif self.state == RobotState.TURN_RIGHT:
                cmd.linear.x = 0.0
                cmd.angular.z = -self.angular_speed
                
            elif self.state == RobotState.WALL_FOLLOW_LEFT:
                # Control proporcional para mantener distancia a la pared
                error = self.wall_distance - self.regions['left']
                cmd.linear.x = self.linear_speed
                cmd.angular.z = self.wall_follow_kp * error
                
            elif self.state == RobotState.WALL_FOLLOW_RIGHT:
                # Control proporcional para mantener distancia a la pared
                error = self.wall_distance - self.regions['right']
                cmd.linear.x = self.linear_speed
                cmd.angular.z = -self.wall_follow_kp * error
                
            elif self.state == RobotState.ROTATE_TO_FIND_PATH:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_speed
        
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        """Bucle principal de control"""
        # Determinar nuevo estado
        new_state = self.determine_state()
        
        # Detectar cambio de estado
        if new_state != self.state:
            self.get_logger().info(f"Estado: {self.state.name} → {new_state.name}")
            self.state = new_state
            self.last_state_change = time.time()
        
        # Ejecutar acción del estado
        self.execute_state()
        
        # Publicar estado de exploración
        exploration_msg = Bool()
        exploration_msg.data = not self.exploration_complete
        self.state_pub.publish(exploration_msg)
        
        # Log periódico de convergencia
        if self.get_clock().now().nanoseconds % 2000000000 < 100000000:  # Cada 2 segundos
            self.get_logger().info(
                f"Convergencia actual: {self.convergence_value:.3f} m "
                f"(objetivo: < {self.convergence_threshold:.3f} m)"
            )

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveExplorer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Asegurar que el robot se detenga
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()