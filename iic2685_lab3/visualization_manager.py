#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from typing import List

# Mensajes ROS2
from geometry_msgs.msg import PoseArray, PointStamped
from std_msgs.msg import Float64, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

class VisualizationManager(Node):
    """
    Nodo para mejorar la visualización del filtro de partículas en RViz
    """
    
    def __init__(self):
        super().__init__('visualization_manager')
        
        # Suscriptores
        self.particles_subscriber = self.create_subscription(
            PoseArray, '/particles', self.particles_callback, 10)
        
        self.best_pose_subscriber = self.create_subscription(
            PointStamped, '/best_pose', self.best_pose_callback, 10)
        
        self.confidence_subscriber = self.create_subscription(
            Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicadores
        self.particles_marker_publisher = self.create_publisher( MarkerArray, '/particles_markers',10)
        
        self.best_pose_marker_publisher = self.create_publisher(
            Marker, '/best_pose_marker', 10)
        
        self.confidence_marker_publisher = self.create_publisher(
            Marker, '/confidence_text', 10)
        
        # Variables
        self.current_confidence = 0.0
        
        self.get_logger().info("Visualization Manager inicializado")
        
    def particles_callback(self, msg: PoseArray):
        """Callback para visualizar partículas como markers"""
        marker_array = MarkerArray()
        
        # Limpiar markers anteriores
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Crear marker para cada partícula
        for i, pose in enumerate(msg.poses):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'particles'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Posición
            marker.pose = pose
            
            # Tamaño
            marker.scale.x = 0.1  # Longitud de la flecha
            marker.scale.y = 0.02  # Ancho de la flecha
            marker.scale.z = 0.02  # Altura de la flecha
            
            # Color azul semi-transparente
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.3
            
            marker_array.markers.append(marker)
        
        self.particles_marker_publisher.publish(marker_array)
        
    def best_pose_callback(self, msg: PointStamped):
        """Callback para visualizar la mejor estimación de pose"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'best_pose'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Posición
        marker.pose.position = msg.point
        marker.pose.orientation.w = 1.0
        
        # Tamaño
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        # Color rojo brillante
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.best_pose_marker_publisher.publish(marker)
        
    def confidence_callback(self, msg: Float64):
        """Callback para mostrar confianza de localización"""
        self.current_confidence = msg.data
        
        # Crear texto marker para mostrar confianza
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'confidence'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Posición del texto (esquina superior del mapa)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 3.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        
        # Texto
        marker.text = f"Confianza: {self.current_confidence:.3f}"
        
        # Tamaño del texto
        marker.scale.z = 0.2
        
        # Color del texto (verde si alta confianza, rojo si baja)
        if self.current_confidence > 0.5:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.confidence_marker_publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    
    visualization_manager = VisualizationManager()
    
    try:
        rclpy.spin(visualization_manager)
    except KeyboardInterrupt:
        pass
    finally:
        visualization_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()