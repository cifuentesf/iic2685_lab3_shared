#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from random import uniform, gauss, choices
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from std_msgs.msg import Float64, Header
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.distance import cdist


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
        self.z_max = 3.5
        
        # Estado del filtro
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.current_scan = None
        self.last_odom = None
        self.prev_odom = None
        
        # Campo de verosimilitud precalculado
        self.likelihood_field = None
        
        # Cargar mapa y inicializar
        self.load_map()
        self.precompute_likelihood_field()
        self.initialize_particles()
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)
        
        # Timer para MCL
        self.create_timer(0.2, self.mcl_update)
        
        self.get_logger().info(f"Filtro MCL iniciado con {self.num_particles} partículas")

    def load_map(self):
        """Cargar mapa desde archivo"""
        try:
            import os
            from ament_index_python.packages import get_package_share_directory
            
            pkg_share = get_package_share_directory('iic2685_lab3')
            map_path = os.path.join(pkg_share, 'maps', 'mapa.pgm')
            
            self.map_data = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            
            # Información del mapa
            self.map_info = {
                'resolution': 0.01,
                'origin': [0.0, 0.0, 0.0],
                'width': self.map_data.shape[1],
                'height': self.map_data.shape[0]
            }
            
            self.get_logger().info(f"Mapa cargado: {self.map_info['width']}x{self.map_info['height']}")
            
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {e}")
            # Mapa dummy
            self.map_data = np.ones((270, 270), dtype=np.uint8) * 255
            self.map_info = {'resolution': 0.01, 'origin': [0.0, 0.0, 0.0], 'width': 270, 'height': 270}

    def precompute_likelihood_field(self):
        """Precalcular campo de verosimilitud - Likelihood Fields Model"""
        self.get_logger().info("Precalculando campo de verosimilitud...")
        
        # Encontrar obstáculos
        obstacle_pixels = np.where(self.map_data < 100)
        
        if len(obstacle_pixels[0]) == 0:
            self.likelihood_field = np.ones_like(self.map_data, dtype=np.float32) * 0.1
            return
        
        # Crear campo de distancias
        h, w = self.map_data.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        all_coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        obstacle_coords = np.column_stack([obstacle_pixels[0], obstacle_pixels[1]])
        
        # Calcular distancias mínimas eficientemente
        distances = cdist(all_coords, obstacle_coords, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Aplicar distribución gaussiana
        likelihood_values = np.exp(-0.5 * (min_distances / (self.sigma_hit / self.map_info['resolution'])) ** 2)
        self.likelihood_field = likelihood_values.reshape(h, w).astype(np.float32)
        
        self.get_logger().info("Campo de verosimilitud calculado")

    def initialize_particles(self):
        """Inicializar partículas en espacio libre"""
        self.particles = []
        attempts = 0
        max_attempts = self.num_particles * 10
        
        while len(self.particles) < self.num_particles and attempts < max_attempts:
            x = uniform(0.5, 2.0)
            y = uniform(0.5, 2.0)
            ang = uniform(-np.pi, np.pi)
            
            if self.is_free_space(x, y):
                particle = Particle(x, y, ang, self.sigma_motion)
                particle.weight = 1.0 / self.num_particles
                self.particles.append(particle)
                
            attempts += 1
        
        # Completar con posiciones aleatorias si es necesario
        while len(self.particles) < self.num_particles:
            x = uniform(0.5, 2.0)
            y = uniform(0.5, 2.0)
            ang = uniform(-np.pi, np.pi)
            particle = Particle(x, y, ang, self.sigma_motion)
            particle.weight = 1.0 / self.num_particles
            self.particles.append(particle)

    def is_free_space(self, x, y):
        """Verificar si posición está en espacio libre"""
        px = int((x - self.map_info['origin'][0]) / self.map_info['resolution'])
        py = int((y - self.map_info['origin'][1]) / self.map_info['resolution'])
        
        if px < 0 or px >= self.map_info['width'] or py < 0 or py >= self.map_info['height']:
            return False
        
        return self.map_data[py, px] > 200

    def laser_callback(self, msg):
        """Callback para datos del LIDAR"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Callback para odometría"""
        self.last_odom = msg

    def mcl_update(self):
        """Algoritmo MCL principal - Tabla 8.2"""
        if self.current_scan is None or len(self.particles) == 0:
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
        ranges = np.clip(ranges, 0.1, self.z_max)
        
        # Submuestrear lecturas para eficiencia
        angle_min = self.current_scan.angle_min
        angle_inc = self.current_scan.angle_increment
        step = max(1, len(ranges) // 20)  # Usar ~20 rayos
        
        for particle in self.particles:
            likelihood = 1.0
            
            for i in range(0, len(ranges), step):
                z = ranges[i]
                if z >= self.z_max:
                    continue
                
                # Calcular punto de impacto esperado
                angle = angle_min + i * angle_inc + particle.ang
                x_hit = particle.x + z * np.cos(angle)
                y_hit = particle.y + z * np.sin(angle)
                
                # Convertir a coordenadas del mapa
                px = int((x_hit - self.map_info['origin'][0]) / self.map_info['resolution'])
                py = int((y_hit - self.map_info['origin'][1]) / self.map_info['resolution'])
                
                # Obtener likelihood del campo precalculado
                if 0 <= px < self.map_info['width'] and 0 <= py < self.map_info['height']:
                    p_hit = self.z_hit * self.likelihood_field[py, px]
                else:
                    p_hit = 0.0
                
                p_random = self.z_random / self.z_max
                likelihood *= (p_hit + p_random)
            
            particle.weight = likelihood

    def resample(self):
        """Remuestreo por importancia"""
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