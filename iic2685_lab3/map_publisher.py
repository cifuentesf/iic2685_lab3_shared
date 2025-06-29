#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, Float32, Bool
import cv2
import yaml
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        
        # Subscriptor
        self.state_sub = self.create_subscription(
            Bool, '/bot_state', self.cb_state, 10)
        self.likelihood_map_sub = self.create_subscription(
            OccupancyGrid, '/map_state', self.cb_map, 10)
        
        # Timer para publicar el mapa periódicamente
        self.timer = self.create_timer(1.0, self.print_map)
        
        self.get_logger().info('Publicador de mapa iniciado')
    
    def cb_map(self, lh_map):
        '''
        Callback del likelihood.
        Activa las fns que permiten generar en terminal el mapa de calor.

        '''
        self.get_logger().info(f'Received message {lh_map}') #Editar

    def cb_state(self, msg_running):
        if not msg_running:
            self.map_result()


    def load_map(self, map_file, yaml_file):
        """Carga el mapa desde archivos PGM y YAML"""
        # Cargar imagen
        self.map_image = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            self.get_logger().error(f'No se pudo cargar el mapa: {map_file}')
            return
        
        # Cargar metadata
        with open(yaml_file, 'r') as f:
            self.map_metadata = yaml.safe_load(f)
        
        self.get_logger().info(f'Mapa cargado: {self.map_image.shape}')
    
    def print_map(self):
        """Publica el mapa como OccupancyGrid"""
        # Crear mensaje OccupancyGrid
        map_msg = OccupancyGrid()
        
        # Header
        map_msg.header = Header()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        
        # Metadata
        map_msg.info.resolution = self.map_metadata['resolution']
        map_msg.info.width = self.map_image.shape[1]
        map_msg.info.height = self.map_image.shape[0]
        
        # Origen
        map_msg.info.origin.position.x = self.map_metadata['origin'][0]
        map_msg.info.origin.position.y = self.map_metadata['origin'][1]
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.x = 0.0
        map_msg.info.origin.orientation.y = 0.0
        map_msg.info.origin.orientation.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # Convertir imagen a datos de ocupación
        # En ROS: 0-100 (0=libre, 100=ocupado, -1=desconocido)
        # En imagen: 0=negro(ocupado), 255=blanco(libre)
        data = []
        for row in reversed(range(self.map_image.shape[0])):  # Invertir filas
            for col in range(self.map_image.shape[1]):
                pixel = self.map_image[row, col]
                if pixel < 128:  # Ocupado
                    data.append(100)
                else:  # Libre
                    data.append(0)
        
        map_msg.data = data
        
        # Publicar
        self.map_pub.publish(map_msg)

    def map_result(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()