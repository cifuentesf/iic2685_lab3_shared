#!/usr/bin/env python3
"""
Actividad 1: Modelo del Sensor - Likelihood Fields
"""
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
import math
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
import os


class LikelihoodFieldsSensorModel(Node):
    def __init__(self):
        super().__init__('sensor_model')
        
        # Parámetros del modelo
        self.declare_parameter('sigma', 0.1)  # Desviación estándar gaussiana
        self.declare_parameter('z_max', 4.0)  # Rango máximo del sensor
        
        self.sigma = self.get_parameter('sigma').get_parameter_value().double_value
        self.z_max = self.get_parameter('z_max').get_parameter_value().double_value
        
        # Variables del mapa
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        self.likelihood_field = None
        self.obstacle_coords = []
        
        # Variables del sensor
        self.current_scan = None
        self.scan_angles = None
        
        # Suscriptores
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info("Suscripciones creadas: /map y /scan")

        # Publicador para likelihood field (para visualización)
        self.likelihood_pub = self.create_publisher(Float64MultiArray, '/likelihood_field', 1)
        self.get_logger().info("Publicador creado: /likelihood_field")

        # Timers
        self.publish_timer = self.create_timer(2.0, self.publish_likelihood_field)
        self.get_logger().info("Sensor model (Likelihood Fields) iniciado")
    
    def map_callback(self, msg):
        """Procesa el mapa recibido y calcula los campos de verosimilitud"""
        self.get_logger().info("Mapa recibido, calculando campos de verosimilitud...")
        
        # Guardar información del mapa
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        
        # Convertir datos del mapa a array 2D
        map_array = np.array(msg.data).reshape((self.map_height, self.map_width))
        
        # Crear mapa binario (0 = libre, 1 = ocupado)
        # OccupancyGrid: -1 = desconocido, 0-100 = probabilidad de ocupación
        occupied_threshold = 65  # Pixels > 65 son considerados ocupados
        self.map_data = (map_array > occupied_threshold).astype(np.uint8)
        
        # Calcular campos de verosimilitud usando la opción recomendada
        self.calculate_likelihood_field_precomputed()
        
        self.get_logger().info("Campos de verosimilitud calculados exitosamente")
    
    def calculate_likelihood_field_precomputed(self):
        """
        Opción recomendada: Precalcular la verosimilitud de cada coordenada
        usando distribución gaussiana centrada en cada obstáculo
        """
        # Encontrar coordenadas de obstáculos
        obstacle_indices = np.where(self.map_data == 1)
        self.obstacle_coords = list(zip(obstacle_indices[1], obstacle_indices[0]))  # (x, y)
        
        if len(self.obstacle_coords) == 0:
            self.get_logger().warn("No se encontraron obstáculos en el mapa")
            self.likelihood_field = np.zeros((self.map_height, self.map_width))
            return
        
        # Crear campo de distancias usando transformada de distancia
        # Invertir el mapa (0 = ocupado, 1 = libre) para la transformada
        free_space = 1 - self.map_data
        distance_field = ndimage.distance_transform_edt(free_space)
        
        # Convertir distancias a coordenadas métricas
        distance_field_metric = distance_field * self.map_resolution
        
        # Calcular likelihood usando distribución gaussiana
        # likelihood = exp(-0.5 * (distance / sigma)^2)
        self.likelihood_field = np.exp(-0.5 * (distance_field_metric / self.sigma) ** 2)
        
        # Normalizar para que la suma sea 1 (opcional)
        self.likelihood_field = self.likelihood_field / np.sum(self.likelihood_field)
        
        self.get_logger().info(f"Campo de verosimilitud calculado. "
                              f"Obstáculos encontrados: {len(self.obstacle_coords)}")
    
    def scan_callback(self, msg):
        """Procesa las lecturas del láser"""
        self.current_scan = msg
        
        # Calcular ángulos de cada rayo
        self.scan_angles = []
        for i in range(len(msg.ranges)):
            angle = msg.angle_min + i * msg.angle_increment
            self.scan_angles.append(angle)
    
    def likelihood_field_range_finder_model(self, z, x, y, theta):
        """
        Implementa el modelo de sensor Likelihood Fields.
            z: distancia medida por el láser [m]
            x, y, theta: pose del robot [m, m, rad]
        Returns:
            probabilidad de la medición
        """
        if self.likelihood_field is None or z > self.z_max:
            return 1e-6  # Probabilidad muy baja
        
        # Calcular punto final del rayo láser en coordenadas del mundo
        z_x = x + z * math.cos(theta)
        z_y = y + z * math.sin(theta)
        
        # Convertir a coordenadas del mapa (píxeles)
        map_x = int((z_x - self.map_origin.position.x) / self.map_resolution)
        map_y = int((z_y - self.map_origin.position.y) / self.map_resolution)
        
        # Verificar límites del mapa
        if (0 <= map_x < self.map_width and 0 <= map_y < self.map_height):
            # El origen está en la esquina inferior izquierda, pero el array
            # tiene (0,0) en la esquina superior izquierda
            array_y = self.map_height - 1 - map_y
            return self.likelihood_field[array_y, map_x]
        else:
            return 1e-6  # Fuera del mapa
    
    def beam_range_finder_model(self, scan, robot_pose):
        """
        Calcula la verosimilitud de toda una lectura láser dada una pose del robot
        Args:
            scan: LaserScan message
            robot_pose: (x, y, theta) pose del robot
        Returns:
            probabilidad total de la lectura
        """
        if self.likelihood_field is None:
            return 1e-6
        
        x, y, theta = robot_pose
        total_likelihood = 1.0
        
        # Procesar cada rayo del láser
        for i, (range_val, beam_angle) in enumerate(zip(scan.ranges, self.scan_angles)):
            if range_val < scan.range_min or range_val > scan.range_max:
                continue  # Ignorar lecturas inválidas
            
            # Ángulo global del rayo
            global_beam_angle = theta + beam_angle
            
            # Calcular likelihood para este rayo
            beam_likelihood = self.likelihood_field_range_finder_model(
                range_val, x, y, global_beam_angle)
            
            # Multiplicar likelihood (en log sería suma)
            total_likelihood *= beam_likelihood
        
        return total_likelihood
    
    def publish_likelihood_field(self):
        """Publica el campo de verosimilitud para visualización"""
        if self.likelihood_field is not None:
            # Aplanar el campo para enviarlo como Float64MultiArray
            flat_field = self.likelihood_field.flatten()
            
            msg = Float64MultiArray()
            msg.data = flat_field.tolist()
            
            self.likelihood_pub.publish(msg)
    
    def visualize_likelihood_field(self, robot_poses=None):
        """
        Crea visualización del campo de verosimilitud
        Args:
            robot_poses: lista de poses [(x, y)] para evaluar
        """
        if self.likelihood_field is None:
            self.get_logger().warn("No hay campo de verosimilitud para visualizar")
            return
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Subplot 1: Mapa original
        ax1.imshow(self.map_data, cmap='gray', origin='lower')
        ax1.set_title('Mapa Original')
        ax1.set_xlabel('X [píxeles]')
        ax1.set_ylabel('Y [píxeles]')
        
        # Subplot 2: Campo de verosimilitud
        im = ax2.imshow(self.likelihood_field, cmap='hot', origin='lower')
        ax2.set_title('Campo de Verosimilitud')
        ax2.set_xlabel('X [píxeles]')
        ax2.set_ylabel('Y [píxeles]')
        plt.colorbar(im, ax=ax2)
        
        # Añadir poses del robot si se proporcionan
        if robot_poses:
            for x, y in robot_poses:
                # Convertir a coordenadas del mapa
                map_x = int((x - self.map_origin.position.x) / self.map_resolution)
                map_y = int((y - self.map_origin.position.y) / self.map_resolution)
                array_y = self.map_height - 1 - map_y
                
                if (0 <= map_x < self.map_width and 0 <= array_y < self.map_height):
                    ax2.plot(map_x, array_y, 'bo', markersize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Guardar figura
        try:
            pkg_share = get_package_share_directory('iic2685_lab3')
            fig_path = os.path.join(pkg_share, 'likelihood_field_visualization.png')
            plt.savefig(fig_path)
            self.get_logger().info(f"Visualización guardada en: {fig_path}")
        except Exception as e:
            self.get_logger().warn(f"No se pudo guardar la visualización: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LikelihoodFieldsSensorModel()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()