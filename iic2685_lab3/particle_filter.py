#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.stats import norm
from random import gauss, uniform
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PointStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64
from tf_transformations import quaternion_from_euler, euler_from_quaternion


class Particle(object):
    """Clase Particle basada en ayudantia_rviz"""
    
    def __init__(self, x, y, ang, sigma=0.1):
        self.x, self.y, self.ang = x, y, ang
        self.last_x, self.last_y, self.last_ang = x, y, ang
        self.sigma = sigma
        self.weight = 1.0  # Extensión para filtro de partículas
        
    def move(self, delta_x, delta_y, delta_ang):
        self.last_x, self.last_y, self.last_ang = self.x, self.y, self.ang
        self.x += delta_x + gauss(0, self.sigma)
        self.y += delta_y + gauss(0, self.sigma)
        self.ang += delta_ang + gauss(0, self.sigma)
        
    def pos(self):
        return [self.x, self.y, self.ang]
        
    def last_pos(self):
        return [self.last_x, self.last_y, self.last_ang]
    
    def move_with_noise(self, dx, dy, dtheta, motion_noise):
        self.last_x, self.last_y, self.last_ang = self.x, self.y, self.ang
        self.x += dx + gauss(0, motion_noise[0])
        self.y += dy + gauss(0, motion_noise[1])
        self.ang += dtheta + gauss(0, motion_noise[2])
        # Normalizar ángulo
        self.ang = np.arctan2(np.sin(self.ang), np.cos(self.ang))
        
    def to_pose(self):
        """Convertir partícula a mensaje Pose para ROS2"""
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, self.ang)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose
    
    def set_weight(self, weight):
        self.weight = weight
        
    @property
    def theta(self):
        return self.ang
    
    @theta.setter
    def theta(self, value):
        self.ang = value


class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parámetros simplificados
        self.num_particles = 500
        self.motion_noise = [0.02, 0.02, 0.01]
        self.max_laser_range = 5.0
        self.convergence_threshold = 0.8
        self.likelihood_sigma = 0.2
        self.min_exploration_time = 10.0
        self.start_time = self.get_clock().now()
        
        # Estado
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.kdtree = None
        self.current_scan = None
        self.previous_odom = None
        self.localized = False
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(OccupancyGrid, '/world_map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)
        
        # Timer
        self.create_timer(0.1, self.update_filter)
        
        self.get_logger().info("Filtro de partículas iniciado")
        
    def map_callback(self, msg):
        """Procesar mapa y crear likelihood field"""
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        # Encontrar obstáculos y crear KDTree
        obstacles = np.argwhere(self.map_data > 50)
        if len(obstacles) > 0:
            obstacles_metric = obstacles * self.map_info.resolution + \
                             [self.map_info.origin.position.y, self.map_info.origin.position.x]
            self.kdtree = KDTree(obstacles_metric)
            self.initialize_particles()
            
    def initialize_particles(self):
        """Inicializar partículas en celdas libres usando estructura de ayudantia_rviz"""
        free_cells = np.argwhere(self.map_data == 0)
        n_free = len(free_cells)
        
        if n_free > 0:
            self.particles = []
            for _ in range(self.num_particles):
                y, x = free_cells[np.random.randint(n_free)]
                mx = self.map_info.origin.position.x + (x + uniform(-0.5, 0.5)) * self.map_info.resolution
                my = self.map_info.origin.position.y + (y + uniform(-0.5, 0.5)) * self.map_info.resolution
                ang = uniform(-np.pi, np.pi)
                
                # Usar constructor de ayudantia_rviz
                particle = Particle(mx, my, ang, sigma=0.02)
                self.particles.append(particle)
                
    def laser_callback(self, msg):
        self.current_scan = msg
        
    def odom_callback(self, msg):
        """Calcular movimiento del robot y actualizar partículas"""
        if self.previous_odom is not None and self.particles:
            # Calcular desplazamiento
            dx = msg.pose.pose.position.x - self.previous_odom.pose.pose.position.x
            dy = msg.pose.pose.position.y - self.previous_odom.pose.pose.position.y
            
            # Cambio de orientación
            _, _, yaw1 = euler_from_quaternion([
                self.previous_odom.pose.pose.orientation.x,
                self.previous_odom.pose.pose.orientation.y,
                self.previous_odom.pose.pose.orientation.z,
                self.previous_odom.pose.pose.orientation.w
            ])
            _, _, yaw2 = euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
            dtheta = np.arctan2(np.sin(yaw2 - yaw1), np.cos(yaw2 - yaw1))
            
            # Mover partículas usando método extendido
            for p in self.particles:
                dx_local = dx * np.cos(p.ang) + dy * np.sin(p.ang)
                dy_local = -dx * np.sin(p.ang) + dy * np.cos(p.ang)
                p.move_with_noise(dx_local, dy_local, dtheta, self.motion_noise)
                
        self.previous_odom = msg
        
    def update_filter(self):
        """Actualización principal del filtro"""
        if not self.particles or self.current_scan is None or self.kdtree is None:
            return
            
        # Solo actualizar pesos si hay movimiento o datos laser nuevos
        self.update_weights()
        self.resample()
        self.publish_state()
        
    def update_weights(self):
        """Actualizar pesos basado en observaciones laser"""
        ranges = np.array(self.current_scan.ranges)
        angle_min = self.current_scan.angle_min
        angle_increment = self.current_scan.angle_increment
        
        # Seleccionar subset de rayos para eficiencia
        step = max(1, len(ranges) // 30)  # Usar máximo 30 rayos
        selected_ranges = ranges[::step]
        selected_angles = np.arange(0, len(ranges), step) * angle_increment + angle_min
        
        for particle in self.particles:
            log_likelihood = 0.0
            
            for i, (range_obs, angle) in enumerate(zip(selected_ranges, selected_angles)):
                if np.isfinite(range_obs) and range_obs < self.max_laser_range:
                    # Calcular punto de impacto esperado
                    global_angle = particle.ang + angle  # Usar .ang en lugar de .theta
                    hit_x = particle.x + range_obs * np.cos(global_angle)
                    hit_y = particle.y + range_obs * np.sin(global_angle)
                    
                    # Distancia al obstáculo más cercano
                    dist, _ = self.kdtree.query([hit_y, hit_x])
                    log_likelihood += norm.logpdf(dist, 0, self.likelihood_sigma)
            
            particle.set_weight(np.exp(log_likelihood))
    
    def resample(self):
        """Resampleo de partículas usando estructura de ayudantia_rviz"""
        weights = np.array([p.weight for p in self.particles])
        
        if np.sum(weights) == 0:
            return
            
        weights /= np.sum(weights)
        
        # Resampleo sistemático
        positions = (np.arange(self.num_particles) + uniform(0, 1)) / self.num_particles
        indexes = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(weights)
        
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        # Crear nuevas partículas basadas en las seleccionadas
        new_particles = []
        for idx in indexes:
            old_p = self.particles[idx]
            # Usar constructor de ayudantia_rviz con ruido para diversidad
            new_p = Particle(
                old_p.x + gauss(0, 0.01),
                old_p.y + gauss(0, 0.01), 
                old_p.ang + gauss(0, 0.02),
                sigma=old_p.sigma
            )
            new_particles.append(new_p)
            
        self.particles = new_particles
    
    def publish_state(self):
        """Publicar estado del filtro"""
        if not self.particles:
            return
            
        # Publicar partículas usando to_pose
        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = [p.to_pose() for p in self.particles]
        self.particles_pub.publish(pose_array)
        
        # Calcular mejor estimación (media ponderada)
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
            
            mean_x = np.sum([p.x * w for p, w in zip(self.particles, weights)])
            mean_y = np.sum([p.y * w for p, w in zip(self.particles, weights)])
            
            # Publicar mejor pose
            best_pose = PointStamped()
            best_pose.header.frame_id = 'map'
            best_pose.header.stamp = self.get_clock().now().to_msg()
            best_pose.point.x = mean_x
            best_pose.point.y = mean_y
            self.best_pose_pub.publish(best_pose)
            
            # Calcular y publicar confianza
            confidence = self.calculate_confidence()
            confidence_msg = Float64()
            confidence_msg.data = confidence
            self.confidence_pub.publish(confidence_msg)
            
            # Verificar localización
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if confidence > self.convergence_threshold and elapsed_time > self.min_exploration_time:
                if not self.localized:
                    self.localized = True
                    self.get_logger().info(f"Robot localizado con confianza: {confidence:.3f}")
    
    def calculate_confidence(self):
        """Calcular confianza basada en dispersión de partículas"""
        if len(self.particles) == 0:
            return 0.0
            
        positions = np.array([[p.x, p.y] for p in self.particles])
        
        # Varianza en posición
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        total_variance = var_x + var_y
        
        # Convertir varianza a confianza (0-1)
        confidence = np.exp(-total_variance * 10)  # Factor de escala
        return min(1.0, confidence)


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