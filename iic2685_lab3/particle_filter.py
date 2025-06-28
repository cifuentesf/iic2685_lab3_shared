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
        super().__init__('particle_filter')
        
        # Parámetros del filtro
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('alpha1', 0.2)  # Ruido en rotación por rotación
        self.declare_parameter('alpha2', 0.2)  # Ruido en rotación por traslación  
        self.declare_parameter('alpha3', 0.2)  # Ruido en traslación por traslación
        self.declare_parameter('alpha4', 0.2)  # Ruido en traslación por rotación
        self.declare_parameter('convergence_threshold', 0.5)  # Umbral de convergencia [m]
        
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.alpha1 = self.get_parameter('alpha1').get_parameter_value().double_value
        self.alpha2 = self.get_parameter('alpha2').get_parameter_value().double_value
        self.alpha3 = self.get_parameter('alpha3').get_parameter_value().double_value
        self.alpha4 = self.get_parameter('alpha4').get_parameter_value().double_value
        self.convergence_threshold = self.get_parameter('convergence_threshold').get_parameter_value().double_value
        
        # Variables del filtro
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.current_scan = None
        self.previous_odom = None
        self.initialized = False
        self.converged = False
        
        # Variables para el modelo de sensor
        self.likelihood_field = None
        self.sensor_model_ready = False
        
        # Suscriptores
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 1)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.particles_pub = self.create_publisher(
            PoseArray, '/particles', 10)
        self.estimated_pose_pub = self.create_publisher(
            PoseStamped, '/estimated_pose', 10)
        self.convergence_pub = self.create_publisher(
            Bool, '/localization_converged', 10)
        self.markers_pub = self.create_publisher(
            MarkerArray, '/particle_markers', 10)
        
        # Timer para publicación de partículas
        self.publish_timer = self.create_timer(0.5, self.publish_particles)
        
        self.get_logger().info(f"Filtro de partículas iniciado con {self.num_particles} partículas")
    
    def map_callback(self, msg):
        """Procesa el mapa y inicializa las partículas"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        if not self.initialized:
            self.initialize_particles()
            self.initialized = True
            self.get_logger().info("Partículas inicializadas")
    
    def initialize_particles(self):
        """Inicializa partículas uniformemente en el espacio libre del mapa"""
        self.particles = []
        
        # Encontrar celdas libres en el mapa
        free_cells = []
        for y in range(self.map_info.height):
            for x in range(self.map_info.width):
                if self.map_data[y, x] == 0:  # Celda libre
                    # Convertir a coordenadas del mundo
                    world_x = self.map_info.origin.position.x + x * self.map_info.resolution
                    world_y = self.map_info.origin.position.y + y * self.map_info.resolution
                    free_cells.append((world_x, world_y))
        
        if len(free_cells) == 0:
            self.get_logger().error("No se encontraron celdas libres en el mapa")
            return
        
        # Crear partículas en posiciones aleatorias
        for i in range(self.num_particles):
            # Seleccionar celda libre aleatoria
            x, y = random.choice(free_cells)
            
            # Orientación aleatoria
            ang = random.uniform(-math.pi, math.pi)
            
            # Crear partícula con sigma para ruido de movimiento
            particle = Particle(x, y, ang, sigma=0.1)
            particle.weight = 1.0 / self.num_particles
            
            self.particles.append(particle)
        
        self.get_logger().info(f"Inicializadas {len(self.particles)} partículas en {len(free_cells)} celdas libres")
    
    def scan_callback(self, msg):
        """Procesa nueva lectura láser y actualiza pesos de partículas"""
        self.current_scan = msg
        
        if self.initialized and not self.converged:
            self.update_particle_weights()
            self.resample_particles()
            self.check_convergence()
    
    def odom_callback(self, msg):
        """Procesa odometría y predice movimiento de partículas"""
        if not self.initialized:
            return
        
        if self.previous_odom is not None:
            # Calcular movimiento desde la última odometría
            delta_x = msg.pose.pose.position.x - self.previous_odom.pose.pose.position.x
            delta_y = msg.pose.pose.position.y - self.previous_odom.pose.pose.position.y
            
            # Obtener orientaciones
            current_orientation = msg.pose.pose.orientation
            prev_orientation = self.previous_odom.pose.pose.orientation
            
            _, _, current_yaw = euler_from_quaternion([
                current_orientation.x, current_orientation.y, 
                current_orientation.z, current_orientation.w])
            _, _, prev_yaw = euler_from_quaternion([
                prev_orientation.x, prev_orientation.y,
                prev_orientation.z, prev_orientation.w])
            
            delta_theta = self.normalize_angle(current_yaw - prev_yaw)
            
            # Aplicar modelo de movimiento a todas las partículas
            if abs(delta_x) > 0.01 or abs(delta_y) > 0.01 or abs(delta_theta) > 0.01:
                self.predict_particles(delta_x, delta_y, delta_theta)
        
        self.previous_odom = msg
    
    def predict_particles(self, delta_x, delta_y, delta_theta):
        """
        Aplica modelo de movimiento con ruido a todas las partículas
        Usando el método move() de la clase Particle de la ayudantía
        """
        for particle in self.particles:
            # Usar el método move() que ya incluye ruido gaussiano
            particle.move(delta_x, delta_y, delta_theta)
    
    def update_particle_weights(self):
        """Actualiza los pesos de las partículas basado en el modelo de sensor"""
        if self.current_scan is None:
            return
        
        total_weight = 0.0
        
        for particle in self.particles:
            # Calcular likelihood usando modelo de sensor
            likelihood = self.beam_range_finder_model(self.current_scan, particle)
            particle.weight = likelihood
            total_weight += likelihood
        
        # Normalizar pesos
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Si todos los pesos son 0, asignar peso uniforme
            uniform_weight = 1.0 / len(self.particles)
            for particle in self.particles:
                particle.weight = uniform_weight
    
    def beam_range_finder_model(self, scan, particle):
        """
        Implementa el modelo de sensor Likelihood Fields para una partícula
        (Versión simplificada - en un sistema real se conectaría con sensor_model.py)
        """
        if self.map_data is None:
            return 1e-6
        
        total_likelihood = 1.0
        
        # Procesar cada rayo del láser (submuestrear para eficiencia)
        step = max(1, len(scan.ranges) // 20)  # Usar solo 20 rayos
        
        for i in range(0, len(scan.ranges), step):
            range_val = scan.ranges[i]
            
            if range_val < scan.range_min or range_val > scan.range_max:
                continue
            
            # Ángulo del rayo
            beam_angle = scan.angle_min + i * scan.angle_increment
            global_beam_angle = particle.ang + beam_angle  # Usar .ang en lugar de .theta
            
            # Punto final del rayo
            end_x = particle.x + range_val * math.cos(global_beam_angle)
            end_y = particle.y + range_val * math.sin(global_beam_angle)
            
            # Convertir a coordenadas del mapa
            map_x = int((end_x - self.map_info.origin.position.x) / self.map_info.resolution)
            map_y = int((end_y - self.map_info.origin.position.y) / self.map_info.resolution)
            
            # Verificar si está dentro del mapa
            if (0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height):
                # Modelo simplificado: mayor likelihood cerca de obstáculos
                if self.map_data[map_y, map_x] > 50:  # Cerca de obstáculo
                    likelihood = 0.8
                elif self.map_data[map_y, map_x] == 0:  # Espacio libre
                    likelihood = 0.1
                else:  # Desconocido
                    likelihood = 0.05
                
                total_likelihood *= likelihood
        
        return max(total_likelihood, 1e-10)  # Evitar likelihood 0
    
    def resample_particles(self):
        """Remuestrea partículas usando rueda de la fortuna ponderada"""
        if len(self.particles) == 0:
            return
        
        # Calcular pesos acumulativos
        weights = [p.weight for p in self.particles]
        
        # Verificar si hay pesos válidos
        if sum(weights) == 0:
            return
        
        # Remuestreo con rueda de la fortuna
        new_particles = []
        
        # Seleccionar partículas según sus pesos
        for _ in range(self.num_particles):
            # Selección estocástica universal
            r = random.uniform(0, sum(weights))
            cumulative_weight = 0
            
            for particle in self.particles:
                cumulative_weight += particle.weight
                if cumulative_weight >= r:
                    # Crear copia de la partícula seleccionada usando constructor de ayudantía
                    new_particle = Particle(particle.x, particle.y, particle.ang, sigma=particle.sigma)
                    new_particle.weight = 1.0/self.num_particles
                    new_particles.append(new_particle)
                    break
        
        self.particles = new_particles
    
    def check_convergence(self):
        """Verifica si el filtro ha convergido"""
        if len(self.particles) == 0:
            return
        
        # Calcular dispersión de las partículas
        positions = [(p.x, p.y) for p in self.particles]
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        
        # Calcular desviación estándar
        variance = sum((p[0] - center_x)**2 + (p[1] - center_y)**2 for p in positions) / len(positions)
        std_dev = math.sqrt(variance)
        
        # Verificar convergencia
        if std_dev < self.convergence_threshold and not self.converged:
            self.converged = True
            self.get_logger().info(f"¡Filtro convergido! Dispersión: {std_dev:.3f} m")
            self.get_logger().info(f"Pose estimada: x={center_x:.3f}, y={center_y:.3f}")
            
            # Publicar convergencia
            convergence_msg = Bool()
            convergence_msg.data = True
            self.convergence_pub.publish(convergence_msg)
    
    def get_estimated_pose(self):
        """Calcula la pose estimada como promedio ponderado de las partículas"""
        if len(self.particles) == 0:
            return None
        
        # Calcular promedio ponderado
        total_weight = sum(p.weight for p in self.particles)
        if total_weight == 0:
            return None
        
        x = sum(p.x * p.weight for p in self.particles) / total_weight
        y = sum(p.y * p.weight for p in self.particles) / total_weight
        
        # Para el ángulo, usar promedio circular
        sin_sum = sum(math.sin(p.ang) * p.weight for p in self.particles) / total_weight
        cos_sum = sum(math.cos(p.ang) * p.weight for p in self.particles) / total_weight
        theta = math.atan2(sin_sum, cos_sum)
        
        return (x, y, theta)
    
    def publish_particles(self):
        """Publica las partículas para visualización en RViz"""
        if len(self.particles) == 0:
            return
        
        # Publicar pose array de partículas
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            pose.position.z = 0.0
            
            # Convertir ángulo a quaternion
            quat = quaternion_from_euler(0, 0, particle.ang)  # Usar .ang en lugar de .theta
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            pose_array.poses.append(pose)
        
        self.particles_pub.publish(pose_array)
        
        # Publicar pose estimada
        estimated_pose = self.get_estimated_pose()
        if estimated_pose:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            
            pose_stamped.pose.position.x = estimated_pose[0]
            pose_stamped.pose.position.y = estimated_pose[1]
            pose_stamped.pose.position.z = 0.0
            
            quat = quaternion_from_euler(0, 0, estimated_pose[2])
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            
            self.estimated_pose_pub.publish(pose_stamped)
    
    def normalize_angle(self, angle):
        """Normaliza un ángulo al rango [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ParticleFilterLocalization()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()