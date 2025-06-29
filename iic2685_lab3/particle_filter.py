#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.stats import norm

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PointStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64
from tf_transformations import quaternion_from_euler, euler_from_quaternion


class Particle:
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        
    def move(self, dx, dy, dtheta, motion_noise):
        """Aplicar modelo de movimiento con ruido Gaussiano"""
        self.x += dx + np.random.normal(0, motion_noise[0])
        self.y += dy + np.random.normal(0, motion_noise[1])
        self.theta += dtheta + np.random.normal(0, motion_noise[2])
        # Normalizar ángulo
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
    def to_pose(self):
        """Convertir partícula a mensaje Pose"""
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, self.theta)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose


class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parámetros internos
        self.num_particles = 1000
        self.sensor_noise = 0.1
        self.motion_noise = [0.02, 0.02, 0.01]
        self.max_laser_range = 5.0
        self.convergence_threshold = 0.85  # Más alto para requerir mayor certeza
        self.likelihood_sigma = 0.3  # Más alto para ser menos sensible
        self.min_exploration_time = 10.0  # Tiempo mínimo antes de declarar localización
        self.start_time = self.get_clock().now()
        
        # Estado
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.likelihood_field = None
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
        
        # Encontrar obstáculos
        obstacles = np.argwhere(self.map_data > 50)
        if len(obstacles) > 0:
            # Convertir a coordenadas métricas
            obstacles_metric = obstacles * self.map_info.resolution + \
                             [self.map_info.origin.position.y, self.map_info.origin.position.x]
            
            # Crear KDTree
            self.kdtree = KDTree(obstacles_metric)
            
            # Calcular likelihood field
            self.calculate_likelihood_field()
            
            # Inicializar partículas
            self.initialize_particles()
            
    def calculate_likelihood_field(self):
        """Precalcular campo de verosimilitud"""
        h, w = self.map_data.shape
        self.likelihood_field = np.zeros((h, w))
        
        # Para cada celda libre
        free_cells = np.argwhere(self.map_data == 0)
        for y, x in free_cells:
            # Coordenadas métricas
            mx = self.map_info.origin.position.x + x * self.map_info.resolution
            my = self.map_info.origin.position.y + y * self.map_info.resolution
            
            # Distancia al obstáculo más cercano
            dist, _ = self.kdtree.query([my, mx])
            
            # Probabilidad gaussiana
            self.likelihood_field[y, x] = norm.pdf(dist, 0, self.likelihood_sigma)
            
    def initialize_particles(self):
        """Inicializar partículas en celdas libres"""
        free_cells = np.argwhere(self.map_data == 0)
        n_free = len(free_cells)
        
        if n_free > 0:
            self.particles = []
            for _ in range(self.num_particles):
                y, x = free_cells[np.random.randint(n_free)]
                
                # Convertir a métricas con ruido
                mx = self.map_info.origin.position.x + (x + np.random.uniform(-0.5, 0.5)) * self.map_info.resolution
                my = self.map_info.origin.position.y + (y + np.random.uniform(-0.5, 0.5)) * self.map_info.resolution
                theta = np.random.uniform(-np.pi, np.pi)
                
                self.particles.append(Particle(mx, my, theta))
                
    def laser_callback(self, msg):
        self.current_scan = msg
        
    def odom_callback(self, msg):
        """Calcular movimiento del robot"""
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
            
            # Mover partículas
            for p in self.particles:
                # Transformar al marco local
                dx_local = dx * np.cos(p.theta) + dy * np.sin(p.theta)
                dy_local = -dx * np.sin(p.theta) + dy * np.cos(p.theta)
                
                p.move(dx_local, dy_local, dtheta, self.motion_noise)
                
        self.previous_odom = msg
        
    def update_filter(self):
        """Actualización principal del filtro"""
        if not self.particles or self.current_scan is None or self.likelihood_field is None:
            return
            
        # Actualizar pesos
        self.update_weights()
        
        # Resamplear
        self.resample()
        
        # Publicar resultados
        self.publish_results()
        
    def update_weights(self):
        """Actualizar pesos usando likelihood field"""
        for p in self.particles:
            weight = 1.0
            
            # Submuestrear rayos láser
            n_beams = len(self.current_scan.ranges)
            for i in range(0, n_beams, max(1, n_beams // 30)):
                r = self.current_scan.ranges[i]
                
                if self.current_scan.range_min < r < self.current_scan.range_max:
                    # Ángulo del rayo
                    angle = self.current_scan.angle_min + i * self.current_scan.angle_increment
                    
                    # Punto final del rayo
                    end_x = p.x + r * np.cos(p.theta + angle)
                    end_y = p.y + r * np.sin(p.theta + angle)
                    
                    # Convertir a coordenadas del mapa
                    map_x = int((end_x - self.map_info.origin.position.x) / self.map_info.resolution)
                    map_y = int((end_y - self.map_info.origin.position.y) / self.map_info.resolution)
                    
                    # Verificar límites y obtener likelihood
                    if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
                        weight *= (self.likelihood_field[map_y, map_x] + 0.1)
                        
            p.weight = weight
            
    def resample(self):
        """Low variance resampling"""
        if not self.particles:
            return
            
        # Normalizar pesos
        weights = np.array([p.weight for p in self.particles])
        weights = weights / weights.sum()
        
        # Resamplear
        new_particles = []
        r = np.random.uniform(0, 1/self.num_particles)
        c = weights[0]
        i = 0
        
        for m in range(self.num_particles):
            u = r + m / self.num_particles
            while u > c and i < len(self.particles) - 1:
                i += 1
                c += weights[i]
                
            p = self.particles[i]
            new_particles.append(Particle(p.x, p.y, p.theta))
            
        self.particles = new_particles
        
    def publish_results(self):
        """Publicar partículas y mejor estimación"""
        # Publicar partículas
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'world_map'
        pose_array.poses = [p.to_pose() for p in self.particles]
        self.particles_pub.publish(pose_array)
        
        # Calcular y publicar mejor pose
        if self.particles:
            # Media de las partículas
            x_mean = np.mean([p.x for p in self.particles])
            y_mean = np.mean([p.y for p in self.particles])
            
            # Media circular para ángulo
            sin_sum = np.mean([np.sin(p.theta) for p in self.particles])
            cos_sum = np.mean([np.cos(p.theta) for p in self.particles])
            theta_mean = np.arctan2(sin_sum, cos_sum)
            
            # Publicar mejor pose
            point = PointStamped()
            point.header = pose_array.header
            point.point.x = x_mean
            point.point.y = y_mean
            self.best_pose_pub.publish(point)
            
            # Calcular y publicar confianza
            std_x = np.std([p.x for p in self.particles])
            std_y = np.std([p.y for p in self.particles])
            # Hacer la confianza más conservadora
            confidence = np.exp(-8 * np.sqrt(std_x**2 + std_y**2))  # Factor aumentado de 5 a 8
            
            # Considerar el tiempo transcurrido
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed_time < self.min_exploration_time:
                confidence *= 0.5  # Reducir confianza si no ha explorado suficiente
            
            conf_msg = Float64()
            conf_msg.data = confidence
            self.confidence_pub.publish(conf_msg)
            
            # Verificar localización
            if confidence > self.convergence_threshold and not self.localized and elapsed_time > self.min_exploration_time:
                self.localized = True
                self.get_logger().info(f"¡Localizado! Pose: ({x_mean:.2f}, {y_mean:.2f}, {theta_mean:.2f})")
                self.get_logger().info(f"Tiempo de exploración: {elapsed_time:.1f}s")
                

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        

if __name__ == '__main__':
    main()