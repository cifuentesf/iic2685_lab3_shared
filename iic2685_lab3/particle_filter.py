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
from random import gauss


class Particle:
    """
    Clase Particle optimizada para el filtro de partículas.
    Basada en la versión de ayudantia_rviz pero mejorada para localización.
    """
    
    def __init__(self, x, y, ang, sigma=0.02):
        """
        Inicializar partícula
        
        Args:
            x (float): Posición x en metros
            y (float): Posición y en metros  
            ang (float): Ángulo en radianes
            sigma (float): Desviación estándar para ruido de movimiento
        """
        self.x = x
        self.y = y
        self.ang = ang
        self.weight = 1.0  # Peso de la partícula para el filtro
        self.sigma = sigma
        
        # Historial para debugging si es necesario
        self.last_x = x
        self.last_y = y
        self.last_ang = ang

    def move(self, delta_x, delta_y, delta_ang):
        """
        Aplicar modelo de movimiento con ruido gaussiano
        
        Args:
            delta_x (float): Cambio en x
            delta_y (float): Cambio en y
            delta_ang (float): Cambio en ángulo
        """
        # Guardar posición anterior
        self.last_x = self.x
        self.last_y = self.y
        self.last_ang = self.ang
        
        # Aplicar movimiento con ruido
        self.x += delta_x + gauss(0, self.sigma)
        self.y += delta_y + gauss(0, self.sigma)
        self.ang += delta_ang + gauss(0, self.sigma * 0.5)  # Menos ruido angular
        
        # Normalizar ángulo
        self.ang = self.normalize_angle(self.ang)

    def pos(self):
        """Retornar posición actual como lista"""
        return [self.x, self.y, self.ang]

    def last_pos(self):
        """Retornar posición anterior como lista"""
        return [self.last_x, self.last_y, self.last_ang]
    
    def distance_to(self, other_particle):
        """
        Calcular distancia euclidiana a otra partícula
        
        Args:
            other_particle (Particle): Otra partícula
            
        Returns:
            float: Distancia euclidiana
        """
        dx = self.x - other_particle.x
        dy = self.y - other_particle.y
        return np.sqrt(dx*dx + dy*dy)
    
    def copy(self):
        """
        Crear copia de la partícula
        
        Returns:
            Particle: Nueva partícula con los mismos valores
        """
        new_particle = Particle(self.x, self.y, self.ang, self.sigma)
        new_particle.weight = self.weight
        new_particle.last_x = self.last_x
        new_particle.last_y = self.last_y
        new_particle.last_ang = self.last_ang
        return new_particle
    
    def add_noise(self, position_noise=None, angle_noise=None):
        """
        Añadir ruido adicional a la partícula
        
        Args:
            position_noise (float): Ruido posicional (usa self.sigma si es None)
            angle_noise (float): Ruido angular (usa self.sigma*0.5 si es None)
        """
        pos_noise = position_noise if position_noise is not None else self.sigma
        ang_noise = angle_noise if angle_noise is not None else self.sigma * 0.5
        
        self.x += gauss(0, pos_noise)
        self.y += gauss(0, pos_noise)  
        self.ang += gauss(0, ang_noise)
        self.ang = self.normalize_angle(self.ang)
    
    @staticmethod
    def normalize_angle(angle):
        """
        Normalizar ángulo al rango [-π, π]
        
        Args:
            angle (float): Ángulo en radianes
            
        Returns:
            float: Ángulo normalizado
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def __str__(self):
        """Representación en string para debugging"""
        return f"Particle(x={self.x:.3f}, y={self.y:.3f}, ang={self.ang:.3f}, w={self.weight:.6f})"
    
    def __repr__(self):
        """Representación para debugging"""
        return self.__str__()

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parámetros del filtro (internos, no externos)
        self.num_particles = 200
        self.sigma_motion = 0.02
        self.sigma_sensor = 0.2
        self.resample_threshold = 0.5
        
        # Partículas y estado
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.current_scan = None
        self.last_odom = None
        self.localized = False
        
        # Cargar mapa
        self.load_map()
        
        # Inicializar partículas distribuidas uniformemente
        self.initialize_particles()
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)
        
        # Timer para el filtro
        self.create_timer(0.2, self.filter_update)
        
        self.get_logger().info(f"Filtro de partículas iniciado con {self.num_particles} partículas")

    def load_map(self):
        """Cargar mapa desde archivo"""
        try:
            import os
            from ament_index_python.packages import get_package_share_directory
            
            pkg_share = get_package_share_directory('iic2685_lab3')
            map_path = os.path.join(pkg_share, 'maps', 'mapa.pgm')
            
            # Cargar imagen del mapa
            self.map_data = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            
            # Información del mapa (hardcoded para simplificar)
            self.map_info = {
                'resolution': 0.01,  # m/pixel
                'origin': [0.0, 0.0, 0.0],  # x, y, yaw
                'width': self.map_data.shape[1],
                'height': self.map_data.shape[0]
            }
            
            self.get_logger().info(f"Mapa cargado: {self.map_info['width']}x{self.map_info['height']} píxeles")
            
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {e}")
            # Crear mapa dummy para pruebas
            self.map_data = np.ones((270, 270), dtype=np.uint8) * 255
            self.map_info = {'resolution': 0.01, 'origin': [0.0, 0.0, 0.0], 'width': 270, 'height': 270}

    def initialize_particles(self):
        """Inicializar partículas uniformemente en espacio libre"""
        self.particles = []
        
        # Dimensiones del mapa en metros
        map_width_m = self.map_info['width'] * self.map_info['resolution']
        map_height_m = self.map_info['height'] * self.map_info['resolution']
        
        for _ in range(self.num_particles):
            # Generar posición aleatoria en espacio libre
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 50:
                x = uniform(0.2, map_width_m - 0.2)
                y = uniform(0.2, map_height_m - 0.2)
                
                if self.is_free_space(x, y):
                    valid_position = True
                else:
                    attempts += 1
            
            # Si no encontramos espacio libre, usar posición por defecto
            if not valid_position:
                x = uniform(0.5, 2.0)
                y = uniform(0.5, 2.0)
            
            ang = uniform(-np.pi, np.pi)
            particle = Particle(x, y, ang, self.sigma_motion)
            self.particles.append(particle)

    def is_free_space(self, x, y):
        """Verificar si una posición está en espacio libre"""
        # Convertir coordenadas del mundo a píxeles
        px = int((x - self.map_info['origin'][0]) / self.map_info['resolution'])
        py = int((y - self.map_info['origin'][1]) / self.map_info['resolution'])
        
        # Verificar límites
        if px < 0 or px >= self.map_info['width'] or py < 0 or py >= self.map_info['height']:
            return False
            
        # En mapas PGM: 255 = libre, 0 = ocupado
        return self.map_data[py, px] > 200

    def laser_callback(self, msg):
        """Callback para datos del LIDAR"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Callback para odometría - aplicar modelo de movimiento"""
        if self.last_odom is not None:
            # Calcular movimiento desde última odometría
            dx = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
            dy = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
            
            # Calcular cambio de orientación
            _, _, current_yaw = euler_from_quaternion([
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            ])
            _, _, last_yaw = euler_from_quaternion([
                self.last_odom.pose.pose.orientation.x, self.last_odom.pose.pose.orientation.y,
                self.last_odom.pose.pose.orientation.z, self.last_odom.pose.pose.orientation.w
            ])
            
            dang = current_yaw - last_yaw
            
            # Aplicar modelo de movimiento a todas las partículas
            if abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dang) > 0.001:
                for particle in self.particles:
                    particle.move(dx, dy, dang)
        
        self.last_odom = msg

    def likelihood_field_model(self, particle, scan):
        """Modelo de sensor usando campos de verosimilitud con 5 sectores optimizado"""
        if scan is None:
            return 1.0
            
        log_likelihood = 0.0
        num_beams = len(scan.ranges)
        
        # Usar 5 sectores para consistencia con navegación
        # Tomar solo algunos rayos de cada sector para eficiencia
        sector_size = num_beams // 5
        
        # Seleccionar rayos representativos de cada sector
        selected_indices = [
            sector_size // 2,          # Right center
            sector_size + sector_size // 2,  # Right-front center  
            2 * sector_size + sector_size // 2,  # Front center
            3 * sector_size + sector_size // 2,  # Left-front center
            4 * sector_size + sector_size // 2   # Left center
        ]
        
        # Añadir algunos rayos adicionales del sector frontal (más importante)
        front_start = 2 * sector_size
        front_end = 3 * sector_size
        front_step = max(1, (front_end - front_start) // 5)
        for i in range(front_start, front_end, front_step):
            if i < num_beams:
                selected_indices.append(i)
        
        for i in selected_indices:
            if i >= num_beams:
                continue
                
            z = scan.ranges[i]
            
            # Ignorar lecturas inválidas
            if not np.isfinite(z) or z <= scan.range_min or z >= scan.range_max:
                continue
                
            # Calcular posición del punto final del rayo
            angle = scan.angle_min + i * scan.angle_increment + particle.ang
            x_hit = particle.x + z * np.cos(angle)
            y_hit = particle.y + z * np.sin(angle)
            
            # Distancia al obstáculo más cercano
            dist_to_obstacle = self.distance_to_nearest_obstacle(x_hit, y_hit)
            
            # Likelihood gaussiano con pesos por sector
            weight = 1.0
            # Dar más peso a mediciones frontales
            if 2 * sector_size <= i < 3 * sector_size:  # Sector frontal
                weight = 2.0
            elif sector_size <= i < 4 * sector_size:  # Sectores frontales laterales
                weight = 1.5
                
            likelihood = np.exp(-0.5 * (dist_to_obstacle / self.sigma_sensor) ** 2)
            log_likelihood += weight * np.log(likelihood + 1e-10)
        
        return np.exp(log_likelihood)

    def distance_to_nearest_obstacle(self, x, y):
        """Distancia al obstáculo más cercano (versión simplificada)"""
        # Convertir a coordenadas de píxel
        px = int((x - self.map_info['origin'][0]) / self.map_info['resolution'])
        py = int((y - self.map_info['origin'][1]) / self.map_info['resolution'])
        
        # Verificar límites
        if px < 0 or px >= self.map_info['width'] or py < 0 or py >= self.map_info['height']:
            return 0.0
            
        # Si estamos en un obstáculo, distancia = 0
        if self.map_data[py, px] < 100:
            return 0.0
            
        # Buscar en una ventana pequeña alrededor del punto
        min_dist = float('inf')
        search_radius = 20  # píxeles
        
        for dx in range(-search_radius, search_radius + 1, 3):
            for dy in range(-search_radius, search_radius + 1, 3):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.map_info['width'] and 0 <= ny < self.map_info['height']:
                    if self.map_data[ny, nx] < 100:  # Obstáculo encontrado
                        dist = np.sqrt(dx**2 + dy**2) * self.map_info['resolution']
                        min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 1.0

    def filter_update(self):
        """Actualización principal del filtro de partículas"""
        if self.current_scan is None or len(self.particles) == 0:
            return
            
        # 1. Actualizar pesos usando modelo de sensor
        total_weight = 0.0
        for particle in self.particles:
            particle.weight = self.likelihood_field_model(particle, self.current_scan)
            total_weight += particle.weight
        
        # Normalizar pesos
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Si todos los pesos son 0, reinicializar
            self.initialize_particles()
            return
        
        # 2. Calcular confianza de localización
        confidence = self.calculate_confidence()
        
        # 3. Remuestreo si es necesario
        if self.effective_sample_size() < self.resample_threshold * self.num_particles:
            self.resample()
        
        # 4. Publicar resultados
        self.publish_particles()
        self.publish_best_pose()
        self.publish_confidence(confidence)

    def effective_sample_size(self):
        """Calcular tamaño efectivo de muestra"""
        sum_weights_squared = sum(p.weight**2 for p in self.particles)
        return 1.0 / sum_weights_squared if sum_weights_squared > 0 else 0

    def calculate_confidence(self):
        """Calcular confianza de localización basada en dispersión de partículas"""
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
        
        # Convertir varianza a confianza (0-1)
        confidence = np.exp(-variance * 10)  # Ajustar factor según necesidad
        return min(1.0, max(0.0, confidence))

    def resample(self):
        """Remuestreo de partículas usando rueda de la fortuna"""
        new_particles = []
        
        # Crear distribución acumulativa
        weights = [p.weight for p in self.particles]
        cumsum = np.cumsum(weights)
        
        for _ in range(self.num_particles):
            r = uniform(0, cumsum[-1])
            idx = np.searchsorted(cumsum, r)
            idx = min(idx, len(self.particles) - 1)
            
            # Copiar partícula seleccionada con pequeña perturbación
            old_particle = self.particles[idx]
            new_particle = Particle(
                old_particle.x + gauss(0, self.sigma_motion),
                old_particle.y + gauss(0, self.sigma_motion),
                old_particle.ang + gauss(0, self.sigma_motion * 0.5),
                self.sigma_motion
            )
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        
        self.particles = new_particles

    def publish_particles(self):
        """Publicar partículas para visualización"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        
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
        point.header.frame_id = 'map'
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
            self.get_logger().info(f"¡Robot localizado con confianza {confidence:.3f}!")


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