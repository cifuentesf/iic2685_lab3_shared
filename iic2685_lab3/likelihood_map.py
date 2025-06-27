#!/usr/bin/env python3
"""
Actividad 1: Modelo del Sensor - Likelihood Fields
Basado en Probabilistic Robotics, Cap. 6 (Thrun et al.)
Este nodo implementa el modelo de sensor Likelihood Fields para localización
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
import numpy as np
import yaml
import os
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import KDTree
from math import cos, sin, pi, sqrt
import threading

class LikelihoodFieldModel(Node):
    def __init__(self):
        super().__init__('likelihood_field_model')
        self.get_logger().info("Nodo de modelo de sensor Likelihood Field iniciado")

        # Parámetros del modelo
        self.sigma_hit = 0.2  # Desviación estándar para P_hit (en metros)
        self.z_max = 4.0      # Rango máximo del LIDAR
        
        # Cargar mapa
        self.map_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'mapas', 'mapa.yaml'
        )
        self.load_map()
        
        # Precalcular campo de verosimilitud
        self.precompute_likelihood_field()
        
        # Suscripciones
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.real_pose_sub = self.create_subscription(
            Pose, '/real_pose', self.real_pose_callback, 10
        )
        
        # Publicadores para visualización
        self.likelihood_viz_pub = self.create_publisher(
            MarkerArray, '/likelihood_field_viz', 10
        )
        
        # Variables de estado
        self.latest_scan = None
        self.real_pose = None
        self.robot_moved = False
        
        # Timer para visualización periódica
        self.create_timer(2.0, self.publish_likelihood_visualization)

    def load_map(self):
        """Carga el mapa desde los archivos yaml y pgm"""
        try:
            with open(self.map_dir, 'r') as f:
                map_metadata = yaml.safe_load(f)

            self.map_resolution = map_metadata['resolution']  # 0.01 m/pixel
            self.map_origin = map_metadata['origin'][:2]      # [x, y] en metros
            
            # Cargar imagen del mapa
            image_path = os.path.join(os.path.dirname(self.map_dir), map_metadata['image'])
            self.occupancy_grid = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if self.occupancy_grid is None:
                self.get_logger().error(f"No se pudo cargar la imagen del mapa: {image_path}")
                return
            
            # Invertir verticalmente (OpenCV vs ROS convention)
            self.occupancy_grid = cv2.flip(self.occupancy_grid, 0)
            
            # Convertir a binario: 0 = ocupado, 1 = libre
            self.binary_map = (self.occupancy_grid > 250).astype(np.uint8)
            self.map_height, self.map_width = self.binary_map.shape
            
            # Encontrar coordenadas de obstáculos para KD-Tree
            obstacle_indices = np.column_stack(np.where(self.binary_map == 0))
            if len(obstacle_indices) > 0:
                self.obstacle_kdtree = KDTree(obstacle_indices)
                self.get_logger().info(
                    f"Mapa cargado: {self.map_width}x{self.map_height} pixels, "
                    f"{len(obstacle_indices)} obstáculos detectados"
                )
            else:
                self.get_logger().error("No se encontraron obstáculos en el mapa")
                
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {str(e)}")

    def precompute_likelihood_field(self):
        """Precalcula el campo de verosimilitud para todo el mapa"""
        self.get_logger().info("Precalculando campo de verosimilitud...")
        
        # Inicializar campo
        self.likelihood_field = np.zeros((self.map_height, self.map_width))
        
        # Convertir sigma de metros a pixels
        sigma_pixels = self.sigma_hit / self.map_resolution
        
        # Para cada celda del mapa
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.binary_map[y, x] == 1:  # Solo para celdas libres
                    # Buscar distancia al obstáculo más cercano
                    dist_pixels, _ = self.obstacle_kdtree.query([y, x])
                    
                    # Aplicar distribución gaussiana
                    likelihood = np.exp(-0.5 * (dist_pixels / sigma_pixels) ** 2)
                    self.likelihood_field[y, x] = likelihood
        
        # Normalizar el campo
        max_val = np.max(self.likelihood_field)
        if max_val > 0:
            self.likelihood_field /= max_val
            
        self.get_logger().info("Campo de verosimilitud precalculado")
        
        # Guardar visualización inicial del campo
        self.save_likelihood_field_image()

    def save_likelihood_field_image(self):
        """Guarda una imagen del campo de verosimilitud"""
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Mapa original
        plt.subplot(1, 2, 1)
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.title('Mapa Original')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.colorbar(label='Valor de ocupación')
        
        # Subplot 2: Campo de verosimilitud
        plt.subplot(1, 2, 2)
        plt.imshow(self.likelihood_field, cmap='hot', origin='lower')
        plt.title('Campo de Verosimilitud P_hit')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.colorbar(label='Verosimilitud')
        
        # Guardar imagen
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/likelihood_field_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Campo de verosimilitud guardado en: {filename}")

    def world_to_map(self, x, y):
        """Convierte coordenadas del mundo (metros) a índices del mapa"""
        mx = int((x - self.map_origin[0]) / self.map_resolution)
        my = int((y - self.map_origin[1]) / self.map_resolution)
        return mx, my

    def map_to_world(self, mx, my):
        """Convierte índices del mapa a coordenadas del mundo (metros)"""
        x = mx * self.map_resolution + self.map_origin[0]
        y = my * self.map_resolution + self.map_origin[1]
        return x, y

    def calculate_pose_likelihood(self, x, y, theta, scan):
        """
        Calcula P(z|x,m) para una pose hipotética usando el modelo Likelihood Fields
        Implementación del algoritmo del libro (Cap 6.4)
        """
        if scan is None:
            return 0.0
        
        # Convertir pose a índices del mapa
        mx, my = self.world_to_map(x, y)
        
        # Verificar límites
        if mx < 0 or mx >= self.map_width or my < 0 or my >= self.map_height:
            return 0.0
        
        # Si la pose está en un obstáculo, probabilidad 0
        if self.binary_map[my, mx] == 0:
            return 0.0
        
        # Calcular probabilidad acumulada
        q = 1.0
        
        # Parámetros del LIDAR según especificación
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        
        # Evaluar un subconjunto de rayos para eficiencia
        step = max(1, len(scan.ranges) // 30)  # Usar ~30 rayos
        
        for i in range(0, len(scan.ranges), step):
            z = scan.ranges[i]
            
            # Ignorar lecturas inválidas
            if z >= self.z_max or np.isnan(z) or np.isinf(z):
                continue
            
            # Ángulo del rayo respecto al robot
            ray_angle = angle_min + i * angle_increment
            
            # Punto final del rayo en coordenadas del mundo
            end_x = x + z * cos(theta + ray_angle)
            end_y = y + z * sin(theta + ray_angle)
            
            # Convertir a índices del mapa
            end_mx, end_my = self.world_to_map(end_x, end_y)
            
            # Verificar límites
            if 0 <= end_mx < self.map_width and 0 <= end_my < self.map_height:
                # Obtener verosimilitud del campo precalculado
                field_val = self.likelihood_field[end_my, end_mx]
                q *= (field_val + 0.1)  # Agregar pequeña constante para evitar ceros
        
        return q

    def scan_callback(self, msg):
        """Callback para datos del LIDAR"""
        self.latest_scan = msg

    def real_pose_callback(self, msg):
        """Callback para la pose real del robot desde el simulador"""
        if self.real_pose is None or self.robot_moved_significantly(msg):
            self.real_pose = msg
            self.robot_moved = True
            self.visualize_likelihood_at_current_pose()

    def robot_moved_significantly(self, new_pose):
        """Verifica si el robot se movió significativamente"""
        if self.real_pose is None:
            return True
        
        dx = new_pose.position.x - self.real_pose.position.x
        dy = new_pose.position.y - self.real_pose.position.y
        
        distance = sqrt(dx*dx + dy*dy)
        return distance > 0.1  # Umbral de 10 cm

    def visualize_likelihood_at_current_pose(self):
        """Genera visualización del campo de verosimilitud para la pose actual"""
        if self.real_pose is None or self.latest_scan is None:
            return
        
        # Crear mapa de calor de verosimilitudes
        likelihood_heatmap = np.zeros((self.map_height, self.map_width))
        
        # Evaluar verosimilitud en una grilla alrededor de la pose actual
        grid_size = 50  # pixels
        step = 2  # evaluar cada 2 pixels
        
        robot_mx, robot_my = self.world_to_map(
            self.real_pose.position.x, 
            self.real_pose.position.y
        )
        
        # Asumiendo orientación 0 para simplificar
        theta = 0.0
        
        for dy in range(-grid_size, grid_size + 1, step):
            for dx in range(-grid_size, grid_size + 1, step):
                mx = robot_mx + dx
                my = robot_my + dy
                
                if 0 <= mx < self.map_width and 0 <= my < self.map_height:
                    x, y = self.map_to_world(mx, my)
                    likelihood = self.calculate_pose_likelihood(x, y, theta, self.latest_scan)
                    likelihood_heatmap[my, mx] = likelihood
        
        # Guardar visualización
        self.save_likelihood_heatmap(likelihood_heatmap, robot_mx, robot_my)

    def save_likelihood_heatmap(self, heatmap, robot_x, robot_y):
        """Guarda el mapa de calor de verosimilitudes con la posición del robot"""
        plt.figure(figsize=(10, 8))
        
        # Mostrar mapa base
        plt.imshow(self.occupancy_grid, cmap='gray', alpha=0.5, origin='lower')
        
        # Superponer mapa de calor
        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
        plt.imshow(masked_heatmap, cmap='hot', alpha=0.7, origin='lower')
        
        # Marcar posición del robot
        plt.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot')
        
        plt.title('Verosimilitud P(z|x,m) alrededor del robot')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.colorbar(label='Verosimilitud')
        plt.legend()
        
        # Guardar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/likelihood_heatmap_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Mapa de calor guardado en: {filename}")

    def publish_likelihood_visualization(self):
        """Publica marcadores para visualización en RViz"""
        if self.latest_scan is None:
            return
        
        marker_array = MarkerArray()
        
        # Crear marcador de grilla
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "likelihood_field"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        
        # Muestrear puntos del campo
        step = 10
        for y in range(0, self.map_height, step):
            for x in range(0, self.map_width, step):
                if self.likelihood_field[y, x] > 0.1:
                    wx, wy = self.map_to_world(x, y)
                    
                    point = Marker()
                    point.header = marker.header
                    point.ns = "likelihood_points"
                    point.id = y * self.map_width + x
                    point.type = Marker.SPHERE
                    point.action = Marker.ADD
                    
                    point.pose.position.x = wx
                    point.pose.position.y = wy
                    point.pose.position.z = 0.0
                    point.pose.orientation.w = 1.0
                    
                    # Escala proporcional a la verosimilitud
                    scale = 0.01 + 0.04 * self.likelihood_field[y, x]
                    point.scale.x = scale
                    point.scale.y = scale
                    point.scale.z = scale
                    
                    # Color según verosimilitud (rojo = alta, azul = baja)
                    point.color.r = self.likelihood_field[y, x]
                    point.color.b = 1.0 - self.likelihood_field[y, x]
                    point.color.g = 0.0
                    point.color.a = 0.5
                    
                    marker_array.markers.append(point)
        
        self.likelihood_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodFieldModel()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()