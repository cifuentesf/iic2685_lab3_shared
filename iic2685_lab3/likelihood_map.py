#!/usr/bin/env python3
# Actividad 1 - Modelo del sensor tipo Likelihood Field
# Basado en Probabilistic Robotics, Cap. 6 (Thrun et al.)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import yaml
import os
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import KDTree
from math import cos, sin, pi

class LikelihoodFieldMap(Node):
    def __init__(self):
        super().__init__('likelihood_map')
        self.get_logger().info("Nodo de mapa de verosimilitud iniciado")

        # Directorio del mapa
        self.map_dir = os.path.join(
            os.path.dirname(__file__), '..', 'mapas', 'mapa.yaml'
        )
        self.load_map()
        
        # Suscripción al LIDAR
        self.sub_lidar = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.scan_received = False
        self.latest_scan = None

    def load_map(self):
        '''Carga el mapa desde mapa.yaml y mapa.pgm'''
        with open(self.map_dir, 'r') as f:
            map_metadata = yaml.safe_load(f)

        self.map_resolution = map_metadata['resolution']
        self.map_origin = map_metadata['origin'][:2]  # [x, y]
        image_path = os.path.join(os.path.dirname(self.map_dir), map_metadata['image'])

        self.occupancy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.occupancy = cv2.flip(self.occupancy, 0)  # PGM empieza desde arriba

        # Celdas libres = 0, ocupadas = 1
        self.binary_map = np.where(self.occupancy < 250, 1, 0)
        self.map_height, self.map_width = self.binary_map.shape

        # Precomputar árbol KD con coordenadas de obstáculos
        obstacle_coords = np.column_stack(np.where(self.binary_map == 1))
        self.kdtree = KDTree(obstacle_coords)

        self.get_logger().info(f"Mapa cargado: {self.map_width}x{self.map_height} celdas")

    def scan_callback(self, msg):
        '''Recibe el escaneo LIDAR y genera el mapa de verosimilitud'''
        if self.scan_received:
            return  # solo uno por ejecución
        self.scan_received = True
        self.latest_scan = msg

        # Parámetros del sensor
        sigma = 3.0  # desviación estándar en celdas (depende de resolución)

        # Evaluar mapa de verosimilitud para orientación θ = 0
        likelihood_map = np.zeros((self.map_height, self.map_width))

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        max_range = msg.range_max

        self.get_logger().info("Generando mapa de verosimilitud...")

        for y in range(0, self.map_height, 5):  # para acelerar
            for x in range(0, self.map_width, 5):
                # Convertir índice (i,j) a coordenada en metros
                wx = x * self.map_resolution + self.map_origin[0]
                wy = y * self.map_resolution + self.map_origin[1]

                prob_sum = 0.0
                for i, z in enumerate(msg.ranges):
                    if z >= max_range or np.isinf(z) or np.isnan(z):
                        continue

                    # Coordenada esperada del haz en el mundo
                    beam_angle = angles[i]
                    zx = wx + z * cos(beam_angle)
                    zy = wy + z * sin(beam_angle)

                    # Convertir a índice de celda
                    mx = int((zx - self.map_origin[0]) / self.map_resolution)
                    my = int((zy - self.map_origin[1]) / self.map_resolution)

                    # Revisar límites
                    if mx < 0 or mx >= self.map_width or my < 0 or my >= self.map_height:
                        continue

                    # Calcular distancia a obstáculo más cercano
                    dist_pix, _ = self.kdtree.query([my, mx])
                    prob = np.exp(-0.5 * (dist_pix / sigma) ** 2)
                    prob_sum += prob

                # Asignar verosimilitud
                likelihood_map[y, x] = prob_sum

        # Normalizar para visualización
        norm_map = (likelihood_map / np.max(likelihood_map)) * 255
        norm_map = norm_map.astype(np.uint8)

        # Guardar imagen
        plt.figure(figsize=(8, 6))
        plt.imshow(norm_map, cmap='hot', origin='lower')
        plt.title("Mapa de Verosimilitud (θ=0)")
        plt.colorbar(label='p(z|x,m)')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("results", exist_ok=True)
        filename = f'results/likelihood_map_{timestamp}.png'
        plt.savefig(filename, dpi=300)
        self.get_logger().info(f"Mapa de verosimilitud guardado en {filename}")
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodFieldMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
