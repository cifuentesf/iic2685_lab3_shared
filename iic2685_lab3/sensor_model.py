#!/usr/bin/env python3
"""
Actividad 1: Modelo del Sensor Likelihood Fields
Implementa el modelo de sensor basado en campos de verosimilitud
como se describe en el capítulo 4.2 del libro y el laboratorio 3.
"""
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import math
from scipy import spatial
from scipy.stats import norm
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Vector3, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from sklearn.metrics import mean_squared_error
from tf_transformations import euler_from_quaternion
from math import pi, atan2, sqrt, cos, sin
import time



class LikelihoodFields(Node):
    def __init__(self):
        super().__init__('likelihood_fields')

        # Parámetros del modelo
        self.sigma_hit = 0.2  # Desviación estándar de la distribución gaussiana
        self.z_max = 4.0  # Rango máximo del sensor
        self.likelihood_field_resolution = 0.01  # Resolución del campo de verosimilitud
        
        # Parametros Robot
        self.robot_pose = None
        self.latest_scan = None
        self.running = True

        # Suscriptores
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.pose_sub = self.create_subscription(Pose, '/real_pose', self.pose_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)


        # Publicadores
        self.likelihood_pub = self.create_publisher(OccupancyGrid, '/map_state', 10)
        self.state_pub = self.create_publisher(Bool, '/bot_state', 10)
        
        # Cargar mapa y precalcular campos de verosimilitud
        self.load_map()
        self.compute_likelihood_field()
        
        self.get_logger().info('Modelo de sensor Likelihood Fields inicializado')

        #Timers
        self.control_timer = self.create_timer(1.0, self.control_loop)

    def load_map(self):
        """Carga el mapa desde los archivos mapa.pgm y mapa.yaml"""
        pkg_share = get_package_share_directory('iic2685_lab3')
        map_path = os.path.join(pkg_share, 'maps', 'mapa.pgm')
        yaml_path = os.path.join(pkg_share, 'maps', 'mapa.yaml')
        
        self.map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            self.get_logger().error(f'No se pudo cargar el mapa: {map_path}')
            return
        
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
        
        self.map_resolution = map_metadata['resolution']  # metros/pixel
        self.map_origin = map_metadata['origin']  # [x, y, theta]
        
        # 0 = obstáculo, 255 = libre
        self.map_data = 255 - self.map_image
        
        # Extraer coordenadas de obstáculos
        self.extract_obstacles()
        
        self.get_logger().info(f'Mapa cargado: {self.map_image.shape}, resolución: {self.map_resolution}')
    

    def compute_likehood_field(self):
        pass
    
    def control_loop(self):
        pass
        if 


    #Callback
    def laser_callback(self, msg):
        """Callback para actualizar la última medición del láser"""
        self.latest_scan = msg
    
    def pose_callback(self, msg):
        """Callback para actualizar la pose del robot"""
        self.current_robot_pose = (msg.position.x, msg.position.y)

    def odom_callback(self, msg):
        '''Callback para odometría del robot'''
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.current_robot_orientation = yaw

class LikelihoodFieldsSensorModel(Node):
    def __init__(self):
        super().__init__('likelihood_fields_sensor_model')

        #Parametros
        self.sigma_hit = 0.2 # Desviación estándar
        self.z_max = 4.0 # Rango máximo del sensor
        self.likelihood_field_resolution = 0.01 # Resolución del campo
        
        # Suscriptores
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.pose_sub = self.create_subscription(Pose, '/real_pose', self.pose_callback, 10)

        # Publicadores
        #self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Estado del robot
        self.robot_pose = None
        self.latest_scan = None
        
        # Fns auxiliares
        # Cargar mapa
        self.load_map()
        # Precalcular campos de verosimilitud
        self.compute_likelihood_field()
        
        # Timer
        self.vis_timer = self.create_timer(1.0, self.visualize_likelihood)
        
        self.get_logger().info('Modelo de sensor Likelihood Fields inicializado')
    
    def load_map(self):
        """Carga el mapa desde los archivos mapa.pgm y mapa.yaml"""
        pkg_share = get_package_share_directory('iic2685_lab3')
        map_path = os.path.join(pkg_share, 'maps', 'mapa.pgm')
        yaml_path = os.path.join(pkg_share, 'maps', 'mapa.yaml')
        
        self.map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            self.get_logger().error(f'No se pudo cargar el mapa: {map_path}')
            return
        
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
        
        self.map_resolution = map_metadata['resolution']  # metros/pixel
        self.map_origin = map_metadata['origin']  # [x, y, theta]
        
        # 0 = obstáculo, 255 = libre
        self.map_data = 255 - self.map_image
        
        # Extraer coordenadas de obstáculos
        self.extract_obstacles()
        
        self.get_logger().info(f'Mapa cargado: {self.map_image.shape}, resolución: {self.map_resolution}')
    
    def extract_obstacles(self):
        """Extrae las coordenadas de los obstáculos del mapa"""
        obstacle_pixels = np.where(self.map_data > 200)
        
        self.obstacle_coords = []
        for i in range(len(obstacle_pixels[0])):
            # Píxeles (fila, columna)
            row = obstacle_pixels[0][i]
            col = obstacle_pixels[1][i]
            
            x = col * self.map_resolution + self.map_origin[0]
            y = (self.map_image.shape[0] - row - 1) * self.map_resolution + self.map_origin[1]
            
            self.obstacle_coords.append([x, y])
        
        self.obstacle_coords = np.array(self.obstacle_coords)
        
        self.obstacle_tree = spatial.KDTree(self.obstacle_coords)
        
        self.get_logger().info(f'Extraídos {len(self.obstacle_coords)} puntos de obstáculos')
    
    def compute_likelihood_field(self):
        """Precalcula el campo de verosimilitud para todo el mapa"""
        self.get_logger().info('Calculando campo de verosimilitud...')
        
        # Crear grid para el campo de verosimilitud
        height, width = self.map_image.shape
        self.likelihood_field = np.zeros((height, width), dtype=np.float32)
        
        # Para cada píxel del mapa
        for row in range(height):
            for col in range(width):
                # Convertir a coordenadas métricas
                x = col * self.map_resolution + self.map_origin[0]
                y = (height - row - 1) * self.map_resolution + self.map_origin[1]
                
                # Encontrar distancia al obstáculo más cercano
                dist, _ = self.obstacle_tree.query([x, y])
                
                # Calcular verosimilitud usando distribución gaussiana
                likelihood = norm.pdf(dist, loc=0, scale=self.sigma_hit)
                self.likelihood_field[row, col] = likelihood
        
        # Normalizar el campo
        max_likelihood = np.max(self.likelihood_field)
        if max_likelihood > 0:
            self.likelihood_field /= max_likelihood
        
        self.get_logger().info('Campo de verosimilitud calculado')
    
    def measurement_model(self, z_t, x_t, map_data=None):
        """
        Implementa P(z_t | x_t, m) usando Likelihood Fields
        
        Args:
            z_t: Medición del láser (LaserScan)
            x_t: Pose hipotética del robot [x, y, theta]
            map_data: Mapa (no usado aquí, pero incluido por compatibilidad)
        
        Returns:
            Probabilidad P(z_t | x_t, m)
        """
        if z_t is None:
            return 0.0
        
        # Probabilidad acumulada
        q = 1.0
        
        # Número de rayos a considerar (subsample para eficiencia)
        step = max(1, len(z_t.ranges) // 30)  # Usar ~30 rayos
        
        for k in range(0, len(z_t.ranges), step):
            if z_t.ranges[k] <= z_t.range_min or z_t.ranges[k] >= z_t.range_max:
                continue
            
            # Ángulo del rayo k
            angle = z_t.angle_min + k * z_t.angle_increment
            
            # Punto final del rayo en coordenadas globales
            x_zk = x_t[0] + z_t.ranges[k] * math.cos(x_t[2] + angle)
            y_zk = x_t[1] + z_t.ranges[k] * math.sin(x_t[2] + angle)
            
            # Buscar distancia al obstáculo más cercano
            dist, _ = self.obstacle_tree.query([x_zk, y_zk])
            
            # Calcular p_hit usando distribución gaussiana
            p_hit = norm.pdf(dist, loc=0, scale=self.sigma_hit)
            
            # Actualizar probabilidad acumulada
            q *= p_hit
        
        return q
    
    def get_likelihood_at_pose(self, x, y, theta=0.0):
        """
        Obtiene la verosimilitud en una pose específica
        Usado para visualización
        """
        if self.latest_scan is None:
            return 0.0
        
        return self.measurement_model(self.latest_scan, [x, y, theta])
    
    def laser_callback(self, msg):
        """Callback para actualizar la última medición del láser"""
        self.latest_scan = msg
    
    def pose_callback(self, msg):
        """Callback para actualizar la pose del robot"""
        # Extraer orientación (asumiendo que es 2D)
        theta = 2 * math.atan2(msg.orientation.z, msg.orientation.w)
        self.robot_pose = [msg.position.x, msg.position.y, theta]
    
    def visualize_likelihood(self):
        """Visualiza el campo de verosimilitud con la pose actual del robot"""
        if self.robot_pose is None or self.latest_scan is None:
            return
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mostrar mapa original
        ax1.imshow(self.map_image, cmap='gray', origin='lower')
        ax1.set_title('Mapa Original')
        ax1.set_xlabel('X (píxeles)')
        ax1.set_ylabel('Y (píxeles)')
        
        # Calcular verosimilitud para la pose actual
        height, width = self.map_image.shape
        likelihood_map = np.zeros((height, width))
        
        # Muestrear poses con ángulo 0
        sample_step = 5  # Muestrear cada 5 píxeles
        for row in range(0, height, sample_step):
            for col in range(0, width, sample_step):
                # Solo calcular para celdas libres
                if self.map_data[row, col] < 100:
                    x = col * self.map_resolution + self.map_origin[0]
                    y = (height - row - 1) * self.map_resolution + self.map_origin[1]
                    
                    likelihood = self.get_likelihood_at_pose(x, y, 0.0)
                    likelihood_map[row, col] = likelihood
        
        # Mostrar campo de verosimilitud
        im = ax2.imshow(likelihood_map, cmap='hot', origin='lower', alpha=0.8)
        ax2.imshow(self.map_image, cmap='gray', origin='lower', alpha=0.3)
        
        # Marcar posición actual del robot
        if self.robot_pose:
            robot_col = int((self.robot_pose[0] - self.map_origin[0]) / self.map_resolution)
            robot_row = height - 1 - int((self.robot_pose[1] - self.map_origin[1]) / self.map_resolution)
            ax2.plot(robot_col, robot_row, 'bo', markersize=10, label='Robot')
        
        ax2.set_title('Campo de Verosimilitud P(z|x,m)')
        ax2.set_xlabel('X (píxeles)')
        ax2.set_ylabel('Y (píxeles)')
        ax2.legend()
        
        plt.colorbar(im, ax=ax2, label='Verosimilitud')
        plt.tight_layout()
        
        # Guardar figura
        save_path = '/tmp/likelihood_field_vis.png'
        plt.savefig(save_path)
        plt.close()
        
        self.get_logger().info(f'Visualización guardada en: {save_path}')


def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodFieldsSensorModel()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()