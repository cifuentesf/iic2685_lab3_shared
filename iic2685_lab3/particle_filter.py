#!/usr/bin/env python3
"""
Actividad 2: Filtro de Partículas para Localización
Implementa Monte Carlo Localization (MCL) usando el modelo de sensor Likelihood Fields
"""
import rclpy
from rclpy.node import Node
import numpy as np
import math
import random
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, Bool
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from random import gauss
import copy


class Particle:
    """
    Representa una partícula en el filtro
    Basada en la clase Particle de la ayudantía
    """
    def __init__(self, x=0.0, y=0.0, ang=0.0, sigma=0.1):
        self.x = x
        self.y = y
        self.ang = ang  
        self.last_x = x
        self.last_y = y
        self.last_ang = ang
        self.sigma = sigma
        self.weight = 1.0  # Peso para el filtro de partículas
    
    def move(self, delta_x, delta_y, delta_ang):
        """
        Mueve la partícula aplicando deltas con ruido gaussiano
        Igual que en la ayudantía
        """
        # Guardar posición anterior
        self.last_x = self.x
        self.last_y = self.y  
        self.last_ang = self.ang
        
        # Movimiento con ruido
        self.x += delta_x + gauss(0, self.sigma)
        self.y += delta_y + gauss(0, self.sigma)
        self.ang += delta_ang + gauss(0, self.sigma)
    
    def pos(self):
        """Retorna posición actual [x, y, ang]"""
        return [self.x, self.y, self.ang]
    
    def last_pos(self):
        """Retorna posición anterior [last_x, last_y, last_ang]"""
        return [self.last_x, self.last_y, self.last_ang]
    
    @property
    def theta(self):
        """Compatibilidad: alias para ang"""
        return self.ang
    
    @theta.setter
    def theta(self, value):
        """Compatibilidad: setter para ang"""
        self.ang = value
    
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f}, ang={self.ang:.2f}, w={self.weight:.4f})"


class ParticleFilterLocalization(Node):
    def __init__(self):
        super().__init__('particle_filter_localization')
        
        # Parámetros del filtro
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('motion_noise_x', 0.05)
        self.declare_parameter('motion_noise_y', 0.05)
        self.declare_parameter('motion_noise_theta', 0.1)
        self.declare_parameter('convergence_threshold', 0.5)
        self.declare_parameter('convergence_percentage', 0.8)
        self.declare_parameter('wall_follow_distance', 0.5)
        self.declare_parameter('forward_speed', 0.2)
        self.declare_parameter('turn_speed', 0.5)
        
        # Obtener parámetros
        self.num_particles = self.get_parameter('num_particles').value
        self.motion_noise = {
            'x': self.get_parameter('motion_noise_x').value,
            'y': self.get_parameter('motion_noise_y').value,
            'theta': self.get_parameter('motion_noise_theta').value
        }
        self.convergence_threshold = self.get_parameter('convergence_threshold').value
        self.convergence_percentage = self.get_parameter('convergence_percentage').value
        self.wall_follow_distance = self.get_parameter('wall_follow_distance').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        
        # Inicializar el modelo de sensor
        from .sensor_model import LikelihoodFieldsSensorModel
        self.sensor_model = LikelihoodFieldsSensorModel()
        
        # Partículas
        self.particles = []
        self.initialize_particles_uniform()
        
        # Estado del robot
        self.current_pose = None
        self.last_odom = None
        self.current_scan = None
        self.localized = False
        self.robot_state = 'LOCALIZING'  # Estados: LOCALIZING, EXPLORING, LOCALIZED
        
        # Suscripciones
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Publicadores
        self.particles_pub = self.create_publisher(
            PoseArray,
            '/particles',  # Compatible con ayudantia_rviz
            10
        )
        
        self.estimated_pose_pub = self.create_publisher(
            PoseStamped,
            '/estimated_pose',
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel_mux/input/navigation',
            10
        )
        
        self.localized_pub = self.create_publisher(
            Bool,
            '/robot_localized',
            10
        )
        
        # Timer para el bucle principal del filtro
        self.create_timer(0.1, self.filter_loop)
        
        # Timer para publicar visualizaciones
        self.create_timer(0.5, self.publish_particles)
        
        self.get_logger().info(f'Filtro de partículas inicializado con {self.num_particles} partículas')
    
    def initialize_particles_uniform(self):
        """Inicializa las partículas uniformemente en el espacio libre del mapa"""
        self.particles = []
        
        # Obtener dimensiones del mapa
        map_width_m = self.sensor_model.map_width * self.sensor_model.map_resolution
        map_height_m = self.sensor_model.map_height * self.sensor_model.map_resolution
        origin_x = self.sensor_model.map_origin.position.x
        origin_y = self.sensor_model.map_origin.position.y
        
        particles_created = 0
        max_attempts = self.num_particles * 10
        attempts = 0
        
        while particles_created < self.num_particles and attempts < max_attempts:
            # Generar posición aleatoria en coordenadas métricas
            x = random.uniform(origin_x, origin_x + map_width_m)
            y = random.uniform(origin_y, origin_y + map_height_m)
            theta = random.uniform(-math.pi, math.pi)
            
            # Verificar si la posición está en espacio libre
            map_x = int((x - origin_x) / self.sensor_model.map_resolution)
            map_y = int((y - origin_y) / self.sensor_model.map_resolution)
            
            if (0 <= map_x < self.sensor_model.map_width and 
                0 <= map_y < self.sensor_model.map_height and
                self.sensor_model.map_data[map_y, map_x] == 0):  # 0 = libre
                
                particle = Particle(x, y, theta, self.motion_noise['x'])
                particle.weight = 1.0 / self.num_particles
                self.particles.append(particle)
                particles_created += 1
            
            attempts += 1
        
        if particles_created < self.num_particles:
            self.get_logger().warn(f"Solo se pudieron crear {particles_created} partículas en espacio libre")
        
        self.normalize_weights()
    
    def odom_callback(self, msg):
        """Callback para la odometría"""
        if self.last_odom is None:
            self.last_odom = msg
            return
        
        # Calcular el movimiento del robot
        dx = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
        
        # Calcular cambio de orientación
        q1 = self.last_odom.pose.pose.orientation
        q2 = msg.pose.pose.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])
        _, _, yaw2 = euler_from_quaternion([q2.x, q2.y, q2.z, q2.w])
        dtheta = yaw2 - yaw1
        
        # Normalizar ángulo
        while dtheta > math.pi:
            dtheta -= 2 * math.pi
        while dtheta < -math.pi:
            dtheta += 2 * math.pi
        
        # Aplicar modelo de movimiento a las partículas
        if abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dtheta) > 0.01:
            self.motion_update(dx, dy, dtheta)
        
        self.last_odom = msg
    
    def scan_callback(self, msg):
        """Callback para el láser"""
        self.current_scan = msg
    
    def motion_update(self, dx, dy, dtheta):
        """
        Actualiza las partículas según el modelo de movimiento
        Compatible con la clase Particle de ayudantia_rviz
        """
        for particle in self.particles:
            # Transformar el movimiento al frame de la partícula
            dx_local = dx * math.cos(particle.ang) + dy * math.sin(particle.ang)
            dy_local = -dx * math.sin(particle.ang) + dy * math.cos(particle.ang)
            
            # Usar el método move de la partícula
            particle.move(dx_local, dy_local, dtheta)
            
            # Normalizar ángulo
            while particle.ang > math.pi:
                particle.ang -= 2 * math.pi
            while particle.ang < -math.pi:
                particle.ang += 2 * math.pi
    
    def measurement_update(self):
        """
        Actualiza los pesos de las partículas basándose en las mediciones del sensor
        """
        if self.current_scan is None:
            return
        
        for particle in self.particles:
            # Calcular verosimilitud usando el modelo de sensor
            likelihood = self.sensor_model.likelihood_field_range_finder_model(
                self.current_scan, 
                particle.x, 
                particle.y, 
                particle.ang
            )
            
            particle.weight *= likelihood
        
        # Normalizar pesos
        self.normalize_weights()
    
    def normalize_weights(self):
        """Normaliza los pesos de las partículas"""
        total_weight = sum(p.weight for p in self.particles)
        
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Si todos los pesos son 0, reinicializar uniformemente
            uniform_weight = 1.0 / len(self.particles)
            for particle in self.particles:
                particle.weight = uniform_weight
    
    def resample(self):
        """
        Remuestreo de partículas usando el algoritmo de la rueda
        """
        if len(self.particles) == 0:
            return
        
        # Crear array de pesos acumulados
        weights = [p.weight for p in self.particles]
        cumulative_weights = np.cumsum(weights)
        
        new_particles = []
        
        # Remuestrear
        for i in range(self.num_particles):
            r = random.random()
            
            # Buscar la partícula correspondiente
            idx = np.searchsorted(cumulative_weights, r)
            
            # Crear nueva partícula (copia)
            old_particle = self.particles[idx]
            new_particle = Particle(
                old_particle.x,
                old_particle.y,
                old_particle.ang,
                self.motion_noise['x']
            )
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def check_convergence(self):
        """
        Verifica si el filtro ha convergido
        """
        if len(self.particles) == 0:
            return False
        
        # Calcular centro de masa de las partículas
        mean_x = sum(p.x * p.weight for p in self.particles)
        mean_y = sum(p.y * p.weight for p in self.particles)
        
        # Calcular cuántas partículas están cerca del centro de masa
        particles_near_mean = 0
        for particle in self.particles:
            distance = math.sqrt((particle.x - mean_x)**2 + (particle.y - mean_y)**2)
            if distance < self.convergence_threshold:
                particles_near_mean += 1
        
        # Verificar si suficientes partículas están cerca
        percentage = particles_near_mean / len(self.particles)
        
        return percentage >= self.convergence_percentage
    
    def get_estimated_pose(self):
        """
        Calcula la pose estimada como el promedio ponderado de las partículas
        """
        if len(self.particles) == 0:
            return None
        
        # Calcular media ponderada
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        
        # Para el ángulo, usar media circular
        sin_sum = sum(p.weight * math.sin(p.ang) for p in self.particles)
        cos_sum = sum(p.weight * math.cos(p.ang) for p in self.particles)
        theta = math.atan2(sin_sum, cos_sum)
        
        return x, y, theta
    
    def wall_following_behavior(self):
        """
        Comportamiento reactivo de seguimiento de pared para exploración
        """
        if self.current_scan is None:
            return
        
        cmd = Twist()
        
        # Obtener lecturas del láser
        ranges = np.array(self.current_scan.ranges)
        
        # Dividir el láser en sectores
        num_readings = len(ranges)
        
        # Sectores: derecha (0°), frente (90°), izquierda (180°)
        right_sector = ranges[0:num_readings//3]
        front_sector = ranges[num_readings//3:2*num_readings//3]
        left_sector = ranges[2*num_readings//3:]
        
        # Calcular distancias mínimas en cada sector
        min_right = np.min(right_sector[np.isfinite(right_sector)])
        min_front = np.min(front_sector[np.isfinite(front_sector)])
        min_left = np.min(left_sector[np.isfinite(left_sector)])
        
        # Lógica de seguimiento de pared (pared a la derecha)
        if min_front < 0.3:  # Obstáculo adelante
            # Girar a la izquierda
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_speed
            self.robot_state = 'TURNING'
        elif min_right > self.wall_follow_distance * 1.2:  # Muy lejos de la pared
            # Girar ligeramente a la derecha
            cmd.linear.x = self.forward_speed * 0.5
            cmd.angular.z = -self.turn_speed * 0.3
            self.robot_state = 'ADJUSTING_RIGHT'
        elif min_right < self.wall_follow_distance * 0.8:  # Muy cerca de la pared
            # Girar ligeramente a la izquierda
            cmd.linear.x = self.forward_speed * 0.5
            cmd.angular.z = self.turn_speed * 0.3
            self.robot_state = 'ADJUSTING_LEFT'
        else:  # Distancia correcta
            # Avanzar recto
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0
            self.robot_state = 'FOLLOWING_WALL'
        
        self.cmd_vel_pub.publish(cmd)
    
    def exploration_step(self):
        """
        Ejecuta un paso de exploración con movimientos discretos
        """
        if self.robot_state == 'EXPLORING':
            # Ejecutar comportamiento de seguimiento de pared
            self.wall_following_behavior()
            
            # Después de moverse, ejecutar actualización del filtro
            if self.current_scan is not None:
                self.measurement_update()
                self.resample()
    
    def filter_loop(self):
        """
        Bucle principal del filtro de partículas
        """
        if not self.localized:
            # Ejecutar filtro de partículas
            if self.current_scan is not None:
                # Actualización de medición
                self.measurement_update()
                
                # Verificar convergencia
                if self.check_convergence():
                    self.localized = True
                    pose = self.get_estimated_pose()
                    if pose:
                        self.get_logger().info(f"¡Robot localizado! Pose: x={pose[0]:.2f}, y={pose[1]:.2f}, theta={pose[2]:.2f}")
                        
                        # Publicar que está localizado
                        msg = Bool()
                        msg.data = True
                        self.localized_pub.publish(msg)
                        
                        # Detener el robot
                        cmd = Twist()
                        self.cmd_vel_pub.publish(cmd)
                else:
                    # Continuar explorando
                    self.exploration_step()
                    
                    # Remuestrear periódicamente
                    if random.random() < 0.1:  # 10% de probabilidad
                        self.resample()
        
        # Publicar pose estimada
        pose = self.get_estimated_pose()
        if pose:
            self.publish_estimated_pose(pose)
    
    def publish_particles(self):
        """
        Publica las partículas para visualización en RViz
        """
        if len(self.particles) == 0:
            return
        
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # Compatible con ayudantia_rviz
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            pose.position.z = 0.0
            
            # Convertir ángulo a quaternion
            q = quaternion_from_euler(0, 0, particle.ang)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
            msg.poses.append(pose)
        
        self.particles_pub.publish(msg)
    
    def publish_estimated_pose(self, pose):
        """
        Publica la pose estimada del robot
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, pose[2])
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        
        self.estimated_pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ParticleFilterLocalization()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Detener el robot antes de salir
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()