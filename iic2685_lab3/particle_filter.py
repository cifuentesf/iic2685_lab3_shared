#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from random import uniform, gauss
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from std_msgs.msg import Float64
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial import cKDTree


class Particle:
    """Clase partícula integrada (basada en ayudantia_rviz)"""
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
        
        # Parámetros del filtro MCL
        self.num_particles = 200
        self.sigma_motion = 0.03        # Ruido de movimiento
        self.sigma_hit = 0.15           # Desviación estándar para likelihood fields
        self.z_hit = 0.95               # Peso para mediciones correctas
        self.z_random = 0.05            # Peso para mediciones aleatorias
        self.z_max = 3.5                # Distancia máxima del sensor
        
        # Partículas y estado
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.current_scan = None
        self.last_odom = None
        self.localized = False
        
        # Campo de verosimilitud (likelihood field) precalculado
        self.likelihood_field = None
        self.obstacle_coordinates = []
        self.obstacle_kdtree = None
        
        # Cargar mapa y precalcular likelihood field
        self.load_map()
        self.precompute_likelihood_field()
        
        # Inicializar partículas
        self.initialize_particles()
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)
        
        # Timer para el filtro MCL
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
                'resolution': 0.01,  # m/pixel
                'origin': [0.0, 0.0, 0.0],
                'width': self.map_data.shape[1],
                'height': self.map_data.shape[0]
            }
            
            self.get_logger().info(f"Mapa cargado: {self.map_info['width']}x{self.map_info['height']} píxeles")
            
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {e}")
            # Crear mapa dummy
            self.map_data = np.ones((270, 270), dtype=np.uint8) * 255
            self.map_info = {'resolution': 0.01, 'origin': [0.0, 0.0, 0.0], 'width': 270, 'height': 270}

    def precompute_likelihood_field(self):
        """Precalcular campo de verosimilitud según Likelihood Fields Model"""
        self.get_logger().info("Precalculando campo de verosimilitud...")
        
        # Encontrar coordenadas de obstáculos
        obstacle_pixels = np.where(self.map_data < 100)  # Píxeles ocupados
        
        if len(obstacle_pixels[0]) == 0:
            # No hay obstáculos, crear campo uniforme
            self.likelihood_field = np.ones_like(self.map_data, dtype=np.float32) * 0.1
            self.get_logger().warn("No se encontraron obstáculos en el mapa")
            return
        
        # Convertir píxeles a coordenadas métricas
        self.obstacle_coordinates = []
        for py, px in zip(obstacle_pixels[0], obstacle_pixels[1]):
            x = px * self.map_info['resolution'] + self.map_info['origin'][0]
            y = py * self.map_info['resolution'] + self.map_info['origin'][1]
            self.obstacle_coordinates.append([x, y])
        
        # Crear KD-Tree para búsqueda rápida de obstáculos más cercanos
        self.obstacle_kdtree = cKDTree(self.obstacle_coordinates)
        
        # Precalcular campo de verosimilitud para cada píxel del mapa
        self.likelihood_field = np.zeros_like(self.map_data, dtype=np.float32)
        
        for py in range(self.map_data.shape[0]):
            for px in range(self.map_data.shape[1]):
                # Convertir píxel a coordenadas métricas
                x = px * self.map_info['resolution'] + self.map_info['origin'][0]
                y = py * self.map_info['resolution'] + self.map_info['origin'][1]
                
                # Encontrar distancia al obstáculo más cercano
                distance, _ = self.obstacle_kdtree.query([x, y])
                
                # Calcular verosimilitud usando distribución gaussiana
                likelihood = np.exp(-0.5 * (distance / self.sigma_hit) ** 2)
                self.likelihood_field[py, px] = likelihood
        
        self.get_logger().info(f"Campo de verosimilitud calculado para {len(self.obstacle_coordinates)} obstáculos")

    def initialize_particles(self):
        """Inicializar partículas uniformemente en espacio libre"""
        self.particles = []
        
        map_width_m = self.map_info['width'] * self.map_info['resolution']
        map_height_m = self.map_info['height'] * self.map_info['resolution']
        
        attempts = 0
        while len(self.particles) < self.num_particles and attempts < self.num_particles * 10:
            x = uniform(0.1, map_width_m - 0.1)
            y = uniform(0.1, map_height_m - 0.1)
            ang = uniform(-np.pi, np.pi)
            
            if self.is_free_space(x, y):
                particle = Particle(x, y, ang, self.sigma_motion)
                particle.weight = 1.0 / self.num_particles
                self.particles.append(particle)
            
            attempts += 1
        
        # Si no se pudieron crear suficientes, llenar con posiciones aleatorias
        while len(self.particles) < self.num_particles:
            x = uniform(0.5, 2.0)
            y = uniform(0.5, 2.0)
            ang = uniform(-np.pi, np.pi)
            particle = Particle(x, y, ang, self.sigma_motion)
            particle.weight = 1.0 / self.num_particles
            self.particles.append(particle)

    def is_free_space(self, x, y):
        """Verificar si una posición está en espacio libre"""
        px = int((x - self.map_info['origin'][0]) / self.map_info['resolution'])
        py = int((y - self.map_info['origin'][1]) / self.map_info['resolution'])
        
        if px < 0 or px >= self.map_info['width'] or py < 0 or py >= self.map_info['height']:
            return False
        
        return self.map_data[py, px] > 200

    def laser_callback(self, msg):
        """Callback para datos del LIDAR"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Callback para odometría - NO aplicar modelo de movimiento aquí"""
        self.last_odom = msg

    def mcl_update(self):
        """Algoritmo MCL principal - Tabla 8.2 del libro"""
        if self.current_scan is None or len(self.particles) == 0:
            return
        
        # Paso 1: Motion Update (Predicción) - solo si hay odometría
        if self.last_odom is not None:
            self.sample_motion_model()
        
        # Paso 2: Measurement Update (Corrección)
        self.measurement_model_update()
        
        # Paso 3: Resampling
        if self.effective_sample_size() < self.num_particles * 0.5:
            self.resample()
        
        # Paso 4: Calcular confianza y publicar resultados
        confidence = self.calculate_confidence()
        self.publish_particles()
        self.publish_best_pose()
        self.publish_confidence(confidence)
        
        # Log de debugging
        if confidence > 0.1:
            max_weight = max(p.weight for p in self.particles)
            avg_weight = sum(p.weight for p in self.particles) / len(self.particles)
            self.get_logger().info(
                f"Confianza: {confidence:.3f} | "
                f"Peso máx: {max_weight:.6f} | "
                f"ESS: {self.effective_sample_size():.1f}",
                throttle_duration_sec=2.0
            )

    def sample_motion_model(self):
        """Aplicar modelo de movimiento a todas las partículas (Línea 4 MCL)"""
        # Movimiento simple: pequeño ruido aleatorio para simular movimiento
        for particle in self.particles:
            # Agregar pequeño ruido de movimiento para diversidad
            dx = gauss(0, self.sigma_motion * 0.1)
            dy = gauss(0, self.sigma_motion * 0.1)
            dang = gauss(0, self.sigma_motion * 0.05)
            
            particle.x += dx
            particle.y += dy
            particle.ang += dang
            particle.ang = Particle.normalize_angle(particle.ang)

    def measurement_model_update(self):
        """Actualizar pesos usando modelo de medición (Línea 5 MCL)"""
        for particle in self.particles:
            particle.weight = self.likelihood_field_measurement_model(particle, self.current_scan)
        
        # Normalizar pesos
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Si todos los pesos son 0, reinicializar uniformemente
            for particle in self.particles:
                particle.weight = 1.0 / self.num_particles

    def likelihood_field_measurement_model(self, particle, scan):
        """Modelo de medición usando Likelihood Fields (Algoritmo del libro)"""
        if scan is None or self.likelihood_field is None:
            return 0.01
        
        # Producto de verosimilitudes para múltiples rayos
        q = 1.0
        
        # Usar subconjunto de rayos para eficiencia
        num_beams = len(scan.ranges)
        step = max(1, num_beams // 20)  # ~20 rayos
        
        for i in range(0, num_beams, step):
            z = scan.ranges[i]
            
            # Ignorar mediciones inválidas
            if not np.isfinite(z) or z <= scan.range_min or z >= scan.range_max:
                continue
            
            # Calcular posición del punto final del rayo
            angle = scan.angle_min + i * scan.angle_increment + particle.ang
            x_hit = particle.x + z * np.cos(angle)
            y_hit = particle.y + z * np.sin(angle)
            
            # Obtener verosimilitud del campo precalculado
            likelihood = self.get_likelihood_from_field(x_hit, y_hit)
            
            # Modelo de medición mixto: hit + random
            p_hit = self.z_hit * likelihood
            p_random = self.z_random / self.z_max
            
            q *= (p_hit + p_random)
        
        return max(q, 1e-10)  # Evitar peso cero

    def get_likelihood_from_field(self, x, y):
        """Obtener verosimilitud del campo precalculado"""
        # Convertir coordenadas métricas a píxeles
        px = int((x - self.map_info['origin'][0]) / self.map_info['resolution'])
        py = int((y - self.map_info['origin'][1]) / self.map_info['resolution'])
        
        # Verificar límites
        if px < 0 or px >= self.map_info['width'] or py < 0 or py >= self.map_info['height']:
            return 0.01
        
        return self.likelihood_field[py, px]

    def resample(self):
        """Remuestreo con reemplazo según pesos (Líneas 8-11 MCL)"""
        new_particles = []
        
        # Crear distribución acumulativa
        weights = [p.weight for p in self.particles]
        cumsum = np.cumsum(weights)
        
        # Remuestreo de rueda de la fortuna
        for _ in range(self.num_particles):
            r = uniform(0, cumsum[-1])
            idx = np.searchsorted(cumsum, r)
            idx = min(idx, len(self.particles) - 1)
            
            # Crear copia de partícula seleccionada con pequeño ruido
            old_particle = self.particles[idx]
            new_particle = Particle(
                old_particle.x + gauss(0, self.sigma_motion * 0.1),
                old_particle.y + gauss(0, self.sigma_motion * 0.1),
                old_particle.ang + gauss(0, self.sigma_motion * 0.05),
                self.sigma_motion
            )
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        
        self.particles = new_particles

    def effective_sample_size(self):
        """Calcular tamaño efectivo de muestra"""
        sum_weights_squared = sum(p.weight**2 for p in self.particles)
        return 1.0 / sum_weights_squared if sum_weights_squared > 0 else 0

    def calculate_confidence(self):
        """Calcular confianza de localización"""
        if len(self.particles) == 0:
            return 0.0
        
        # Calcular centroide ponderado
        sum_x = sum(p.x * p.weight for p in self.particles)
        sum_y = sum(p.y * p.weight for p in self.particles)
        sum_w = sum(p.weight for p in self.particles)
        
        if sum_w == 0:
            return 0.0
        
        centroid_x = sum_x / sum_w
        centroid_y = sum_y / sum_w
        
        # Calcular dispersión
        variance = sum(p.weight * ((p.x - centroid_x)**2 + (p.y - centroid_y)**2) for p in self.particles) / sum_w
        
        # Convertir varianza a confianza
        confidence = np.exp(-variance * 20)  # Factor ajustado para convergencia
        return min(1.0, max(0.0, confidence))

    def publish_particles(self):
        """Publicar partículas para visualización"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'odom'
        
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
        
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'odom'
        point.point.x = best_particle.x
        point.point.y = best_particle.y
        point.point.z = 0.0
        
        self.best_pose_pub.publish(point)

    def publish_confidence(self, confidence):
        """Publicar confianza de localización"""
        conf_msg = Float64()
        conf_msg.data = confidence
        self.confidence_pub.publish(conf_msg)
        
        # Log cuando se alcanza alta confianza
        if confidence > 0.8 and not self.localized:
            self.localized = True
            best_particle = max(self.particles, key=lambda p: p.weight)
            self.get_logger().info(
                f"¡Robot localizado! Confianza: {confidence:.3f} | "
                f"Pose: ({best_particle.x:.3f}, {best_particle.y:.3f}, {best_particle.ang:.3f}rad)"
            )


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