#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from random import uniform, gauss, choices
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from std_msgs.msg import Float64, Header
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.distance import cdist
import yaml
import os


class Particle:
    """Clase partícula basada en ayudantia_rviz"""
    def __init__(self, x, y, ang, sigma=0.02):
        self.x, self.y, self.ang = x, y, ang
        self.weight = 1.0
        self.sigma = sigma

    def move(self, delta_x, delta_y, delta_ang):
        """Aplicar modelo de movimiento con ruido gaussiano"""
        self.x += delta_x + gauss(0, self.sigma)
        self.y += delta_y + gauss(0, self.sigma)
        self.ang += delta_ang + gauss(0, self.sigma * 0.5)
        self.ang = self.normalize_angle(self.ang)

    def pos(self):
        return [self.x, self.y, self.ang]
    
    @staticmethod
    def normalize_angle(angle):
        """Normalizar ángulo al rango [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parámetros del filtro MCL (internos)
        self.num_particles = 150
        self.sigma_motion = 0.03
        self.sigma_hit = 0.15
        self.z_hit = 0.95
        self.z_random = 0.05
        self.z_max = 4.0  # RANGO MÁXIMO 4 METROS
        
        # Estado del filtro
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.likelihood_field = None
        self.last_odom = None
        self.prev_odom = None
        self.current_scan = None
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        # Publishers - CORRECCIÓN: Cambiar Pose por PointStamped
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)  # ← CORREGIDO
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)
        
        # Timer para actualización MCL
        self.create_timer(0.1, self.mcl_update)
        
        # Cargar mapa
        self.load_map()
        
        self.get_logger().info("Filtro MCL iniciado")

    def load_map(self):
        """Cargar mapa desde archivo yaml"""
        try:
            # Buscar archivo de mapa en el paquete
            map_path = "/home/mcifuentesf/ros2_ws/install/iic2685_lab3/share/iic2685_lab3/maps/mapa.yaml"
            
            with open(map_path, 'r') as file:
                map_yaml = yaml.safe_load(file)
            
            # Cargar imagen del mapa
            image_path = os.path.join(os.path.dirname(map_path), map_yaml['image'])
            self.map_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if self.map_data is None:
                self.get_logger().error(f"No se pudo cargar la imagen del mapa: {image_path}")
                return
            
            # Información del mapa
            self.map_resolution = map_yaml['resolution']
            self.map_origin = map_yaml['origin']
            
            self.get_logger().info(f"Mapa cargado: {self.map_data.shape[1]}x{self.map_data.shape[0]}")
            
            # Precalcular campo de verosimilitud
            self.get_logger().info("Precalculando campo de verosimilitud...")
            self.precompute_likelihood_field()
            
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {e}")

    def precompute_likelihood_field(self):
        """Precalcular campo de verosimilitud usando transformada de distancia"""
        if self.map_data is None:
            return
        
        # Binarizar mapa (0 = obstáculo, 255 = libre)
        binary_map = (self.map_data > 127).astype(np.uint8) * 255
        
        # Calcular transformada de distancia
        dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        
        # Convertir a campo de verosimilitud gaussiano
        self.likelihood_field = np.exp(-0.5 * (dist_transform / self.sigma_hit) ** 2)
        
        self.get_logger().info("Campo de verosimilitud calculado")
        
        # Inicializar partículas uniformemente en el mapa
        self.initialize_particles()

    def initialize_particles(self):
        """Inicializar partículas uniformemente en espacios libres"""
        if self.map_data is None:
            return
        
        self.particles = []
        
        # Encontrar píxeles libres (valores > 127 en escala de grises)
        free_pixels = np.where(self.map_data > 127)
        
        for _ in range(self.num_particles):
            # Seleccionar píxel libre aleatorio
            idx = np.random.randint(0, len(free_pixels[0]))
            map_y, map_x = free_pixels[0][idx], free_pixels[1][idx]
            
            # Convertir a coordenadas del mundo
            world_x = map_x * self.map_resolution + self.map_origin[0]
            world_y = (self.map_data.shape[0] - map_y) * self.map_resolution + self.map_origin[1]
            
            # Orientación aleatoria
            theta = uniform(-np.pi, np.pi)
            
            particle = Particle(world_x, world_y, theta, self.sigma_motion)
            particle.weight = 1.0 / self.num_particles
            self.particles.append(particle)
        
        self.get_logger().info(f"Filtro MCL iniciado con {self.num_particles} partículas")

    def scan_callback(self, msg):
        """Callback para datos del láser"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Callback para odometría"""
        self.last_odom = msg

    def map_callback(self, msg):
        """Callback para mapa (si se recibe por tópico)"""
        pass  # Ya cargamos el mapa desde archivo

    def mcl_update(self):
        """Actualización principal del filtro MCL"""
        if (self.map_data is None or self.likelihood_field is None or 
            self.current_scan is None or len(self.particles) == 0):
            return
        
        # 1. Motion Update (Predicción)
        if self.last_odom is not None:
            self.sample_motion_model()
        
        # 2. Measurement Update (Corrección)
        self.measurement_model_update()
        
        # 3. Resampling
        if self.effective_sample_size() < self.num_particles * 0.5:
            self.resample()
        
        # 4. Publicar resultados
        confidence = self.calculate_confidence()
        self.publish_particles()
        self.publish_best_pose()
        self.publish_confidence(confidence)

    def sample_motion_model(self):
        """Modelo de movimiento simple"""
        if self.prev_odom is None:
            self.prev_odom = self.last_odom
            return
        
        # Calcular movimiento desde odometría
        dx = self.last_odom.pose.pose.position.x - self.prev_odom.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - self.prev_odom.pose.pose.position.y
        
        curr_yaw = euler_from_quaternion([
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
            self.last_odom.pose.pose.orientation.w
        ])[2]
        
        prev_yaw = euler_from_quaternion([
            self.prev_odom.pose.pose.orientation.x,
            self.prev_odom.pose.pose.orientation.y,
            self.prev_odom.pose.pose.orientation.z,
            self.prev_odom.pose.pose.orientation.w
        ])[2]
        
        dyaw = Particle.normalize_angle(curr_yaw - prev_yaw)
        
        # Aplicar movimiento a todas las partículas
        for particle in self.particles:
            particle.move(dx, dy, dyaw)
        
        self.prev_odom = self.last_odom

    def measurement_model_update(self):
        """Actualización con modelo de sensor - Likelihood Fields"""
        ranges = np.array(self.current_scan.ranges)
        ranges[~np.isfinite(ranges)] = self.z_max
        ranges[ranges <= 0.0] = self.z_max
        ranges[ranges > self.z_max] = self.z_max
        
        angle_min = self.current_scan.angle_min
        angle_increment = self.current_scan.angle_increment
        
        # Submuestrear lecturas para eficiencia
        step = max(1, len(ranges) // 36)  # ~36 rayos
        
        for particle in self.particles:
            log_likelihood = 0.0
            
            for i in range(0, len(ranges), step):
                if not np.isfinite(ranges[i]) or ranges[i] <= 0.0 or ranges[i] >= self.z_max:
                    continue
                
                # Ángulo del rayo en el marco del robot
                ray_angle = angle_min + i * angle_increment
                
                # Punto final del rayo en coordenadas del mundo
                end_x = particle.x + ranges[i] * np.cos(particle.ang + ray_angle)
                end_y = particle.y + ranges[i] * np.sin(particle.ang + ray_angle)
                
                # Convertir a coordenadas del mapa
                map_x = int((end_x - self.map_origin[0]) / self.map_resolution)
                map_y = int((self.map_data.shape[0] - (end_y - self.map_origin[1]) / self.map_resolution))
                
                # Verificar límites
                if (0 <= map_x < self.likelihood_field.shape[1] and 
                    0 <= map_y < self.likelihood_field.shape[0]):
                    
                    # Obtener verosimilitud del campo precalculado
                    likelihood = self.likelihood_field[map_y, map_x]
                    log_likelihood += np.log(self.z_hit * likelihood + self.z_random / self.z_max)
            
            # Actualizar peso de la partícula
            particle.weight = np.exp(log_likelihood)

    def resample(self):
        """Remuestreo de baja varianza"""
        if len(self.particles) == 0:
            return
        
        weights = np.array([p.weight for p in self.particles])
        
        if np.sum(weights) == 0:
            weights = np.ones(len(weights))
        
        weights /= np.sum(weights)
        
        # Remuestreo con reemplazo
        indices = choices(range(len(self.particles)), weights=weights, k=self.num_particles)
        new_particles = []
        
        for i in indices:
            old_particle = self.particles[i]
            new_particle = Particle(old_particle.x, old_particle.y, old_particle.ang, self.sigma_motion)
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        
        self.particles = new_particles

    def effective_sample_size(self):
        """Calcular tamaño efectivo de muestra"""
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0:
            return self.num_particles
        weights /= np.sum(weights)
        return 1.0 / np.sum(weights ** 2)

    def calculate_confidence(self):
        """Calcular confianza de localización"""
        if len(self.particles) == 0:
            return 0.0
        
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0:
            return 0.0
        
        weights /= np.sum(weights)
        max_weight = np.max(weights)
        
        return min(max_weight * 10, 1.0)

    def publish_particles(self):
        """Publicar partículas para visualización"""
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.frame_id = "odom"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            pose.position.z = 0.0
            
            quat = quaternion_from_euler(0, 0, particle.ang)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            pose_array.poses.append(pose)
        
        self.particles_pub.publish(pose_array)

    def publish_best_pose(self):
        """Publicar mejor estimación de pose"""
        if len(self.particles) == 0:
            return
        
        # Encontrar partícula con mayor peso
        best_particle = max(self.particles, key=lambda p: p.weight)
        
        # CORRECCIÓN: Mantener PointStamped ya que el publisher ahora es correcto
        point_msg = PointStamped()
        point_msg.header.frame_id = "odom"
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = best_particle.x
        point_msg.point.y = best_particle.y
        point_msg.point.z = 0.0
        
        self.best_pose_pub.publish(point_msg)

    def publish_confidence(self, confidence):
        """Publicar confianza de localización"""
        msg = Float64()
        msg.data = confidence
        self.confidence_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    particle_filter = ParticleFilter()
    
    try:
        rclpy.spin(particle_filter)
    except KeyboardInterrupt:
        pass
    finally:
        particle_filter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()