#!/usr/bin/env python3
"""
Modelo del Sensor - Likelihood Fields
Implementación del modelo de sensor basado en campos de verosimilitud para localización
"""
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.stats import norm
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
import math


class SensorModel(Node):
    def __init__(self):
        super().__init__('sensor_model')
        
        # Parámetros del modelo
        self.declare_parameter('sigma_hit', 0.2)  # Desviación estándar para Phit
        self.declare_parameter('z_max', 5.0)     # Rango máximo del sensor
        self.declare_parameter('likelihood_field_resolution', 0.05)  # Resolución del campo
        
        self.sigma_hit = self.get_parameter('sigma_hit').get_parameter_value().double_value
        self.z_max = self.get_parameter('z_max').get_parameter_value().double_value
        self.resolution = self.get_parameter('likelihood_field_resolution').get_parameter_value().double_value
        
        # Variables del mapa
        self.map_data = None
        self.map_info = None
        self.obstacles = []
        self.kd_tree = None
        self.likelihood_field = None
        
        # Suscriptores
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 1)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/test_pose', self.test_pose_callback, 10)
        
        # Publicadores
        self.likelihood_map_pub = self.create_publisher(
            OccupancyGrid, '/likelihood_map', 10)
        self.markers_pub = self.create_publisher(
            MarkerArray, '/likelihood_markers', 10)
        
        self.get_logger().info("Modelo del sensor iniciado")
    
    def map_callback(self, msg):
        """Callback para recibir el mapa y calcular el campo de verosimilitud"""
        self.get_logger().info("Mapa recibido, calculando campo de verosimilitud...")
        
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width))
        
        # Extraer coordenadas de obstáculos
        self.extract_obstacles()
        
        # Construir KD-Tree para búsqueda eficiente
        if self.obstacles:
            self.kd_tree = KDTree(self.obstacles)
            
            # Pre-calcular campo de verosimilitud
            self.compute_likelihood_field()
            
            # Publicar campo de verosimilitud como mapa
            self.publish_likelihood_map()
            
            self.get_logger().info("Campo de verosimilitud calculado exitosamente")
        else:
            self.get_logger().warn("No se encontraron obstáculos en el mapa")
    
    def extract_obstacles(self):
        """Extrae las coordenadas métricas de los obstáculos del mapa"""
        self.obstacles = []
        
        for i in range(self.map_info.height):
            for j in range(self.map_info.width):
                # Si la celda está ocupada (valor > 65)
                if self.map_data[i, j] > 65:
                    # Convertir a coordenadas métricas
                    x = j * self.map_info.resolution + self.map_info.origin.position.x
                    y = i * self.map_info.resolution + self.map_info.origin.position.y
                    self.obstacles.append([x, y])
    
    def compute_likelihood_field(self):
        """Pre-calcula el campo de verosimilitud para todo el mapa"""
        # Crear grid de verosimilitud
        width = int(self.map_info.width * self.map_info.resolution / self.resolution)
        height = int(self.map_info.height * self.map_info.resolution / self.resolution)
        
        self.likelihood_field = np.zeros((height, width))
        
        # Para cada celda del campo
        for i in range(height):
            for j in range(width):
                # Coordenadas métricas
                x = j * self.resolution + self.map_info.origin.position.x
                y = i * self.resolution + self.map_info.origin.position.y
                
                # Encontrar distancia al obstáculo más cercano
                if self.kd_tree:
                    dist, _ = self.kd_tree.query([x, y])
                    
                    # Calcular probabilidad usando distribución normal
                    prob = norm.pdf(dist, loc=0, scale=self.sigma_hit)
                    self.likelihood_field[i, j] = prob
    
    def sensor_model_likelihood_field(self, z_t, x_t, map_data):
        """
        Calcula P(z_t | x_t, m) usando el modelo de likelihood fields
        
        Args:
            z_t: Medición del láser (LaserScan)
            x_t: Pose hipotética del robot [x, y, theta]
            map_data: Mapa del entorno
        
        Returns:
            q: Verosimilitud de la medición
        """
        if self.kd_tree is None or z_t is None:
            return 0.0
        
        q = 1.0
        
        # Para cada rayo del láser
        for k, z_k in enumerate(z_t.ranges):
            # Ignorar mediciones inválidas
            if z_k >= self.z_max or z_k <= z_t.range_min or math.isnan(z_k):
                continue
            
            # Calcular ángulo del rayo
            angle = z_t.angle_min + k * z_t.angle_increment
            
            # Posición del punto final del rayo
            x_zk = x_t[0] + z_k * math.cos(x_t[2] + angle)
            y_zk = x_t[1] + z_k * math.sin(x_t[2] + angle)
            
            # Encontrar distancia al obstáculo más cercano
            dist, _ = self.kd_tree.query([x_zk, y_zk])
            
            # Calcular probabilidad
            prob = norm.pdf(dist, loc=0, scale=self.sigma_hit)
            q *= prob
        
        return q
    
    def scan_callback(self, msg):
        """Callback para recibir datos del láser"""
        # Guardar para uso posterior
        self.current_scan = msg
    
    def test_pose_callback(self, msg):
        """Callback para probar el modelo con una pose específica"""
        if self.current_scan and self.kd_tree:
            pose = [msg.pose.position.x, 
                   msg.pose.position.y,
                   self.get_yaw_from_quaternion(msg.pose.orientation)]
            
            likelihood = self.sensor_model_likelihood_field(
                self.current_scan, pose, self.map_data)
            
            self.get_logger().info(f"Verosimilitud en pose ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}): {likelihood:.6f}")
    
    def get_yaw_from_quaternion(self, q):
        """Extrae el ángulo yaw de un quaternion"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def publish_likelihood_map(self):
        """Publica el campo de verosimilitud como un OccupancyGrid para visualización"""
        if self.likelihood_field is None:
            return
        
        # Normalizar el campo para visualización
        field_normalized = (self.likelihood_field / np.max(self.likelihood_field) * 100).astype(int)
        
        # Crear mensaje OccupancyGrid
        likelihood_map = OccupancyGrid()
        likelihood_map.header.frame_id = "map"
        likelihood_map.header.stamp = self.get_clock().now().to_msg()
        
        # Copiar info del mapa original pero ajustar resolución
        likelihood_map.info = self.map_info
        likelihood_map.info.resolution = self.resolution
        likelihood_map.info.width = self.likelihood_field.shape[1]
        likelihood_map.info.height = self.likelihood_field.shape[0]
        
        # Aplanar y convertir a lista
        likelihood_map.data = field_normalized.flatten().tolist()
        
        self.likelihood_map_pub.publish(likelihood_map)
    
    def visualize_likelihood_at_poses(self, poses):
        """
        Visualiza la verosimilitud en un conjunto de poses hipotéticas
        Para cumplir con el entregable de la actividad 1
        """
        if self.current_scan is None or self.kd_tree is None:
            return
        
        markers = MarkerArray()
        
        for i, (x, y) in enumerate(poses):
            # Calcular verosimilitud para pose con theta=0
            pose = [x, y, 0.0]
            likelihood = self.sensor_model_likelihood_field(
                self.current_scan, pose, self.map_data)
            
            # Crear marcador
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            
            marker.scale.x = self.map_info.resolution
            marker.scale.y = self.map_info.resolution
            marker.scale.z = 0.1
            
            # Color basado en verosimilitud
            marker.color.a = 0.5
            marker.color.r = likelihood * 100
            marker.color.g = 0.0
            marker.color.b = 1.0 - (likelihood * 100)
            
            markers.markers.append(marker)
        
        self.markers_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    
    sensor_model = SensorModel()
    
    try:
        rclpy.spin(sensor_model)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_model.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()