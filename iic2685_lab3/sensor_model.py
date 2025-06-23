#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import cv2
import yaml
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class SensorModel(Node):
    def __init__(self):
        super().__init__('sensor_model')
        
        # Parámetros del modelo
        self.sigma_hit = 0.1  # Desviación estándar para P_hit
        self.z_max = 4.0      # Distancia máxima del LIDAR
        self.map_resolution = 0.01  # metros por pixel
        self.map_origin = [0.0, 0.0, 0.0]
        
        # Variables del mapa
        self.occupancy_grid = None
        self.likelihood_field = None
        self.obstacle_coords = []
        
        # Suscriptores
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        # Publisher para visualización
        self.pose_array_pub = self.create_publisher(PoseArray, '/likelihood_poses', 10)
        
        self.get_logger().info("Modelo de sensor inicializado")
    
    def map_callback(self, msg):
        """Procesa el mapa recibido y calcula el campo de verosimilitud"""
        self.get_logger().info("Mapa recibido, calculando campo de verosimilitud...")
        
        # Convertir OccupancyGrid a imagen
        width = msg.info.width
        height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, 
                          msg.info.origin.position.y, 
                          msg.info.origin.position.z]
        
        # Crear imagen del mapa
        map_data = np.array(msg.data).reshape((height, width))
        
        # Convertir a formato de imagen (0-255)
        self.occupancy_grid = np.zeros((height, width), dtype=np.uint8)
        self.occupancy_grid[map_data == 100] = 0    # Ocupado = negro
        self.occupancy_grid[map_data == 0] = 255    # Libre = blanco
        self.occupancy_grid[map_data == -1] = 128   # Desconocido = gris
        
        # Calcular campo de verosimilitud
        self.calculate_likelihood_field()
        
        self.get_logger().info("Campo de verosimilitud calculado")
    
    def calculate_likelihood_field(self):
        """Calcula el campo de verosimilitud usando distribución gaussiana"""
        height, width = self.occupancy_grid.shape
        
        # Encontrar coordenadas de obstáculos
        obstacle_pixels = np.where(self.occupancy_grid == 0)  # Pixels negros = obstáculos
        self.obstacle_coords = list(zip(obstacle_pixels[1], obstacle_pixels[0]))  # (x, y)
        
        # Crear KD-Tree para búsqueda eficiente
        if len(self.obstacle_coords) > 0:
            tree = spatial.KDTree(self.obstacle_coords)
            
            # Calcular distancia mínima para cada punto del mapa
            self.likelihood_field = np.zeros((height, width))
            
            for y in range(height):
                for x in range(width):
                    if self.occupancy_grid[y, x] != 0:  # Solo para espacios libres
                        dist, _ = tree.query([x, y])
                        # Convertir distancia de pixels a metros
                        dist_meters = dist * self.map_resolution
                        # Aplicar distribución gaussiana
                        likelihood = np.exp(-0.5 * (dist_meters / self.sigma_hit) ** 2)
                        self.likelihood_field[y, x] = likelihood
        
        self.get_logger().info(f"Campo calculado con {len(self.obstacle_coords)} obstáculos")
    
    def scan_callback(self, msg):
        """Procesa los datos del LIDAR"""
        if self.likelihood_field is None:
            return
        
        # Procesar scan (ejemplo con pose fija)
        robot_x, robot_y, robot_theta = 2.0, 2.0, 0.0  # Pose de ejemplo
        likelihood = self.calculate_likelihood(msg, robot_x, robot_y, robot_theta)
        
        self.get_logger().info(f"Verosimilitud calculada: {likelihood:.6f}")
    
    def calculate_likelihood(self, scan, x, y, theta):
        """Calcula la verosimilitud P(z_t | x_t, m) para una pose dada"""
        if self.likelihood_field is None:
            return 0.0
        
        likelihood = 1.0
        
        # Parámetros del LIDAR (según especificación)
        angle_min = -np.pi/2  # -90°
        angle_max = np.pi/2   # 90°
        angle_increment = (angle_max - angle_min) / len(scan.ranges)
        
        for i, range_reading in enumerate(scan.ranges):
            # Saltar lecturas inválidas
            if range_reading >= self.z_max or range_reading <= 0:
                continue
            
            # Calcular ángulo del rayo
            ray_angle = angle_min + i * angle_increment + theta
            
            # Calcular posición del punto detectado
            point_x = x + range_reading * np.cos(ray_angle)
            point_y = y + range_reading * np.sin(ray_angle)
            
            # Convertir a coordenadas del mapa
            map_x = int((point_x - self.map_origin[0]) / self.map_resolution)
            map_y = int((point_y - self.map_origin[1]) / self.map_resolution)
            
            # Verificar límites del mapa
            height, width = self.likelihood_field.shape
            if 0 <= map_x < width and 0 <= map_y < height:
                # Obtener verosimilitud del campo precalculado
                point_likelihood = self.likelihood_field[map_y, map_x]
                likelihood *= point_likelihood
        
        return likelihood
    
    def visualize_likelihood_field(self, robot_poses=None):
        """Visualiza el campo de verosimilitud"""
        if self.likelihood_field is None:
            self.get_logger().warn("Campo de verosimilitud no disponible")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Crear colormap personalizado
        colors = ['black', 'blue', 'green', 'yellow', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('likelihood', colors, N=n_bins)
        
        # Mostrar mapa base
        plt.subplot(1, 2, 1)
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.title('Mapa Original')
        plt.colorbar()
        
        # Mostrar campo de verosimilitud
        plt.subplot(1, 2, 2)
        plt.imshow(self.likelihood_field, cmap=cmap, origin='lower')
        plt.title('Campo de Verosimilitud')
        plt.colorbar()
        
        # Agregar posiciones del robot si se proporcionan
        if robot_poses:
            for pose in robot_poses:
                map_x = int((pose[0] - self.map_origin[0]) / self.map_resolution)
                map_y = int((pose[1] - self.map_origin[1]) / self.map_resolution)
                plt.plot(map_x, map_y, 'ro', markersize=8)
        
        plt.tight_layout()
        plt.show()
    
    def test_likelihood_calculation(self):
        """Función de prueba para calcular verosimilitudes en diferentes poses"""
        if self.likelihood_field is None:
            self.get_logger().warn("Esperando mapa...")
            return
        
        # Poses de prueba (x, y, theta)
        test_poses = [
            (1.0, 1.0, 0.0),
            (2.0, 2.0, 0.0),
            (3.0, 1.5, 0.0),
            (1.5, 3.0, 0.0)
        ]
        
        # Simular datos de LIDAR (ejemplo)
        fake_scan = LaserScan()
        fake_scan.ranges = [1.0, 1.5, 2.0, 0.8, 1.2] * 36  # 180 lecturas
        
        likelihoods = []
        for pose in test_poses:
            likelihood = self.calculate_likelihood(fake_scan, pose[0], pose[1], pose[2])
            likelihoods.append(likelihood)
            self.get_logger().info(
                f"Pose ({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}): "
                f"Verosimilitud = {likelihood:.6f}"
            )
        
        return test_poses, likelihoods

def main(args=None):
    rclpy.init(args=args)
    sensor_model = SensorModel()
    
    try:
        rclpy.spin(sensor_model)
    except KeyboardInterrupt:
        pass
    
    sensor_model.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()