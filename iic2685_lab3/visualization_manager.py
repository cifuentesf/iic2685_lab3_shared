#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseArray, PointStamped
from std_msgs.msg import Float64, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class VisualizationManager(Node):
    def __init__(self):
        super().__init__('visualization_manager')
        
        # Suscriptores
        self.create_subscription(PoseArray, '/particles', self.particles_callback, 10)
        self.create_subscription(PointStamped, '/best_pose', self.best_pose_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicadores
        self.particles_marker_pub = self.create_publisher(MarkerArray, '/particles_markers', 10)
        self.best_pose_marker_pub = self.create_publisher(Marker, '/best_pose_marker', 10)
        self.confidence_text_pub = self.create_publisher(Marker, '/confidence_text', 10)
        
        # Estado
        self.current_confidence = 0.0
        
        self.get_logger().info("Gestor de visualización iniciado")
        
    def particles_callback(self, msg):
        """Visualizar partículas como markers"""
        marker_array = MarkerArray()
        
        # Limpiar markers anteriores
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = 'particles'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Crear markers para subset de partículas (para eficiencia)
        step = max(1, len(msg.poses) // 200)  # Mostrar máximo 200 partículas
        
        for i, pose in enumerate(msg.poses[::step]):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'particles'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Posición y orientación
            marker.pose = pose
            
            # Tamaño
            marker.scale.x = 0.1  # Longitud flecha
            marker.scale.y = 0.02  # Ancho flecha
            marker.scale.z = 0.02  # Alto flecha
            
            # Color (azul transparente)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.6
            
            marker_array.markers.append(marker)
            
        self.particles_marker_pub.publish(marker_array)
        
    def best_pose_callback(self, msg):
        """Visualizar mejor estimación de pose"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'best_pose'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Posición
        marker.pose.position = msg.point
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        
        # Tamaño
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.2
        
        # Color (verde si alta confianza, amarillo si media, rojo si baja)
        if self.current_confidence > 0.7:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif self.current_confidence > 0.4:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        marker.color.a = 0.8
        
        self.best_pose_marker_pub.publish(marker)
        
    def confidence_callback(self, msg):
        """Mostrar confianza como texto"""
        self.current_confidence = msg.data
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'confidence'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Posición del texto
        marker.pose.position.x = 0.0
        marker.pose.position.y = 3.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        
        # Texto con estado
        status = "EXPLORANDO"
        if self.current_confidence > 0.8:
            status = "LOCALIZADO"
        elif self.current_confidence > 0.5:
            status = "CONVERGIENDO"
            
        marker.text = f"Confianza: {self.current_confidence:.3f}\nEstado: {status}"
        
        # Tamaño
        marker.scale.z = 0.15
        
        # Color según confianza
        if self.current_confidence > 0.7:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif self.current_confidence > 0.4:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        marker.color.a = 1.0
        
        self.confidence_text_pub.publish(marker)


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