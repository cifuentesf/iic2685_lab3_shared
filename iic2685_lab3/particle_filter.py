#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, Twist, PoseWithCovarianceStamped
import numpy as np
from scipy import spatial
import random

class ParticleFilterLocalization(Node):
    def __init__(self):
        super().__init__('particle_filter_localization')
        
        # Parámetros del filtro de partículas
        self.num_particles = 500
        self.particles = None
        self.weights = None
        
        # Parámetros del modelo de movimiento
        self.motion_noise_linear = 0.1   # Ruido en movimiento lineal
        self.motion_noise_angular = 0.05  # Ruido en movimiento angular
        
        # Parámetros del modelo de sensor
        self.sigma_hit = 0.1
        self.z_max = 4.0
        self.map_resolution = 0.01
        self.map_origin = [0.0, 0.0, 0.0]
        
        # Variables del mapa
        self.occupancy_grid = None
        self.likelihood_field = None
        self.obstacle_coords = []
        
        # Variables de control
        self.last_scan = None
        self.localized = False
        self.convergence_threshold = 0.5  # Threshold para determinar convergencia
        
        # Suscriptores
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/robot_pose', 10)
        
        # Timer para navegación reactiva
        self.navigation_timer = self.create_timer(0.5, self.reactive_navigation)
        
        self.get_logger().info("Filtro de partículas inicializado")
    
    def map_callback(self, msg):
        """Procesa el mapa y inicializa las partículas"""
        self.get_logger().info("Mapa recibido, inicializando filtro...")
        
        # Guardar información del mapa
        width = msg.info.width
        height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, 
                          msg.info.origin.position.y, 
                          msg.info.origin.position.z]
        
        # Convertir mapa
        map_data = np.array(msg.data).reshape((height, width))
        self.occupancy_grid = np.zeros((height, width), dtype=np.uint8)
        self.occupancy_grid[map_data == 100] = 0    # Ocupado
        self.occupancy_grid[map_data == 0] = 255    # Libre
        self.occupancy_grid[map_data == -1] = 128   # Desconocido
        
        # Calcular campo de verosimilitud
        self.calculate_likelihood_field()
        
        # Inicializar partículas
        self.initialize_particles()
        
        self.get_logger().info("Filtro inicializado correctamente")
    
    def calculate_likelihood_field(self):
        """Calcula el campo de verosimilitud"""
        height, width = self.occupancy_grid.shape
        
        # Encontrar obstáculos
        obstacle_pixels = np.where(self.occupancy_grid == 0)
        self.obstacle_coords = list(zip(obstacle_pixels[1], obstacle_pixels[0]))
        
        if len(self.obstacle_coords) > 0:
            tree = spatial.KDTree(self.obstacle_coords)
            self.likelihood_field = np.zeros((height, width))
            
            for y in range(height):
                for x in range(width):
                    if self.occupancy_grid[y, x] != 0:  # Solo espacios libres
                        dist, _ = tree.query([x, y])
                        dist_meters = dist * self.map_resolution
                        likelihood = np.exp(-0.5 * (dist_meters / self.sigma_hit) ** 2)
                        self.likelihood_field[y, x] = likelihood
    
    def initialize_particles(self):
        """Inicializa las partículas uniformemente en el espacio libre"""
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Encontrar espacios libres
        free_spaces = np.where(self.occupancy_grid == 255)
        free_coords = list(zip(free_spaces[1], free_spaces[0]))  # (x, y) en pixels
        
        if len(free_coords) == 0:
            self.get_logger().error("No se encontraron espacios libres en el mapa")
            return
        
        # Generar partículas aleatorias en espacios libres
        for _ in range(self.num_particles):
            # Seleccionar posición aleatoria en espacio libre
            pixel_coord = random.choice(free_coords)
            
            # Convertir a coordenadas métricas
            x = pixel_coord[0] * self.map_resolution + self.map_origin[0]
            y = pixel_coord[1] * self.map_resolution + self.map_origin[1]
            theta = random.uniform(-np.pi, np.pi)
            
            self.particles.append([x, y, theta])
        
        self.particles = np.array(self.particles)
        self.get_logger().info(f"Inicializadas {self.num_particles} partículas")
    
    def motion_model(self, particle, linear_vel, angular_vel, dt):
        """Modelo de movimiento con ruido gaussiano"""
        x, y, theta = particle
        
        # Agregar ruido al movimiento
        noisy_linear = linear_vel + np.random.normal(0, self.motion_noise_linear)
        noisy_angular = angular_vel + np.random.normal(0, self.motion_noise_angular)
        
        # Actualizar pose
        new_theta = theta + noisy_angular * dt
        new_x = x + noisy_linear * np.cos(new_theta) * dt
        new_y = y + noisy_linear * np.sin(new_theta) * dt
        
        return [new_x, new_y, new_theta]
    
    def sensor_model(self, particle, scan):
        """Calcula la verosimilitud de una partícula dado el scan"""
        if self.likelihood_field is None:
            return 1.0
        
        x, y, theta = particle
        likelihood = 1.0
        
        # Parámetros del LIDAR
        angle_min = -np.pi/2
        angle_max = np.pi/2
        angle_increment = (angle_max - angle_min) / len(scan.ranges)
        
        # Tomar solo algunas lecturas para eficiencia (cada 10 rayos)
        for i in range(0, len(scan.ranges), 10):
            range_reading = scan.ranges[i]
            
            if range_reading >= self.z_max or range_reading <= 0:
                continue
            
            # Calcular posición del punto detectado
            ray_angle = angle_min + i * angle_increment + theta
            point_x = x + range_reading * np.cos(ray_angle)
            point_y = y + range_reading * np.sin(ray_angle)
            
            # Convertir a coordenadas del mapa
            map_x = int((point_x - self.map_origin[0]) / self.map_resolution)
            map_y = int((point_y - self.map_origin[1]) / self.map_resolution)
            
            # Verificar límites y obtener verosimilitud
            height, width = self.likelihood_field.shape
            if 0 <= map_x < width and 0 <= map_y < height:
                point_likelihood = self.likelihood_field[map_y, map_x]
                likelihood *= (point_likelihood + 0.01)  # Evitar likelihood 0
        
        return likelihood
    
    def scan_callback(self, msg):
        """Procesa el scan del LIDAR y actualiza el filtro"""
        if self.particles is None:
            return
        
        self.last_scan = msg
        
        # Paso de actualización (measurement update)
        for i in range(self.num_particles):
            self.weights[i] = self.sensor_model(self.particles[i], msg)
        
        # Normalizar pesos
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Remuestreo si es necesario
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        if effective_sample_size < self.num_particles / 2:
            self.resample()
        
        # Verificar convergencia
        self.check_convergence()
        
        # Publicar partículas
        self.publish_particles()
        
        # Publicar pose estimada
        self.publish_estimated_pose()
    
    def resample(self):
        """Remuestreo de partículas basado en los pesos"""
        # Remuestreo sistemático
        indices = np.random.choice(
            self.num_particles, 
            self.num_particles, 
            p=self.weights
        )
        
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def check_convergence(self):
        """Verifica si el filtro ha convergido"""
        if self.particles is None:
            return
        
        # Calcular desviación estándar de las posiciones
        std_x = np.std(self.particles[:, 0])
        std_y = np.std(self.particles[:, 1])
        
        if std_x < self.convergence_threshold and std_y < self.convergence_threshold:
            if not self.localized:
                self.localized = True
                # Calcular pose estimada
                mean_x = np.mean(self.particles[:, 0])
                mean_y = np.mean(self.particles[:, 1])
                mean_theta = np.mean(self.particles[:, 2])
                
                self.get_logger().info(
                    f"¡ROBOT LOCALIZADO! Pose: ({mean_x:.2f}, {mean_y:.2f}, {mean_theta:.2f})"
                )
                
                # Detener el robot
                self.stop_robot()
    
    def reactive_navigation(self):
        """Navegación reactiva para exploración"""
        if self.localized or self.last_scan is None:
            return
        
        cmd = Twist()
        
        # Obtener lecturas del frente, izquierda y derecha
        scan_ranges = np.array(self.last_scan.ranges)
        valid_ranges = scan_ranges[scan_ranges < self.z_max]
        
        if len(valid_ranges) == 0:
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # Índices para frente, izquierda y derecha (aproximados)
            front_idx = len(scan_ranges) // 2
            left_idx = 3 * len(scan_ranges) // 4
            right_idx = len(scan_ranges) // 4
            
            front_dist = scan_ranges[front_idx] if scan_ranges[front_idx] < self.z_max else self.z_max
            left_dist = scan_ranges[left_idx] if scan_ranges[left_idx] < self.z_max else self.z_max
            right_dist = scan_ranges[right_idx] if scan_ranges[right_idx] < self.z_max else self.z_max
            
            # Lógica de navegación reactiva
            if front_dist > 1.0:  # Camino libre al frente
                cmd.linear.x = 0.3
                # Seguir pared derecha
                if right_dist < 0.8:
                    cmd.angular.z = 0.2  # Girar ligeramente a la izquierda
                elif right_dist > 1.2:
                    cmd.angular.z = -0.2  # Girar ligeramente a la derecha
            else:  # Obstáculo al frente
                cmd.linear.x = 0.0
                # Decidir dirección de giro
                if left_dist > right_dist:
                    cmd.angular.z = 0.5  # Girar a la izquierda
                else:
                    cmd.angular.z = -0.5  # Girar a la derecha
        
        # Actualizar partículas con el movimiento
        if self.particles is not None:
            dt = 0.5  # Período del timer
            for i in range(self.num_particles):
                self.particles[i] = self.motion_model(
                    self.particles[i], 
                    cmd.linear.x, 
                    cmd.angular.z, 
                    dt
                )
        
        self.cmd_vel_pub.publish(cmd)
    
    def stop_robot(self):
        """Detiene el robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def publish_particles(self):
        """Publica las partículas para visualización"""
        if self.particles is None:
            return
        
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0.0
            
            # Convertir ángulo a quaternion
            pose.orientation.z = np.sin(particle[2] / 2.0)
            pose.orientation.w = np.cos(particle[2] / 2.0)
            
            pose_array.poses.append(pose)
        
        self.particles_pub.publish(pose_array)
    
    def publish_estimated_pose(self):
        """Publica la pose estimada del robot"""
        if self.particles is None:
            return
        
        # Calcular pose promedio ponderada
        weighted_x = np.sum(self.particles[:, 0] * self.weights)
        weighted_y = np.sum(self.particles[:, 1] * self.weights)
        weighted_theta = np.sum(self.particles[:, 2] * self.weights)
        
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.pose.position.x = weighted_x
        pose_msg.pose.pose.position.y = weighted_y
        pose_msg.pose.pose.position.z = 0.0
        
        pose_msg.pose.pose.orientation.z = np.sin(weighted_theta / 2.0)
        pose_msg.pose.pose.orientation.w = np.cos(weighted_theta / 2.0)
        
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    pf_localization = ParticleFilterLocalization()
    
    try:
        rclpy.spin(pf_localization)
    except KeyboardInterrupt:
        pass
    
    pf_localization.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()