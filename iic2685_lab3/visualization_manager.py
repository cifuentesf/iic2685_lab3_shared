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
        self.particle_count = 0
        
        self.get_logger().info("Gestor de visualización iniciado")
        
    def particles_callback(self, msg):
        """Visualizar partículas como markers individuales"""
        marker_array = MarkerArray()
        
        # Primero, limpiar markers anteriores
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = 'particles'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Crear markers para cada partícula (limitar a 100 para eficiencia)
        step = max(1, len(msg.poses) // 100)
        
        for i, pose in enumerate(msg.poses[::step]):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'particles'
            marker.id = i + 1  # ID > 0 para evitar conflicto con DELETE
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Copiar pose completa
            marker.pose = pose
            
            # Tamaño visible pero no demasiado grande
            marker.scale.x = 0.08  # Longitud flecha
            marker.scale.y = 0.015  # Ancho flecha
            marker.scale.z = 0.015  # Alto flecha
            
            # Color azul semi-transparente
            marker.color.r = 0.0
            marker.color.g = 0.2
            marker.color.b = 1.0
            marker.color.a = 0.7
            
            # Tiempo de vida para auto-limpieza
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            
            marker_array.markers.append(marker)
        
        self.particle_count = len(msg.poses)
        self.particles_marker_pub.publish(marker_array)
        
        # Log ocasional para debugging
        if self.particle_count > 0 and (self.particle_count % 100 == 0):
            self.get_logger().info(f"Visualizando {len(marker_array.markers)-1} partículas de {self.particle_count}")
        
    def best_pose_callback(self, msg):
        """Visualizar mejor estimación de pose"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'best_pose'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Posición desde el PointStamped
        marker.pose.position.x = msg.point.x
        marker.pose.position.y = msg.point.y
        marker.pose.position.z = 0.05  # Elevar un poco
        marker.pose.orientation.w = 1.0
        
        # Tamaño del cilindro
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1
        
        # Color según confianza
        if self.current_confidence > 0.7:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0  # Verde
        elif self.current_confidence > 0.4:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0  # Amarillo
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0  # Rojo
        marker.color.a = 0.8
        
        # Tiempo de vida
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        
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
        
        # Posición del texto (ajustar según tu mapa)
        marker.pose.position.x = 0.5
        marker.pose.position.y = 3.0
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        # Determinar estado
        if self.current_confidence > 0.8:
            status = "LOCALIZADO ✓"
        elif self.current_confidence > 0.5:
            status = "CONVERGIENDO..."
        else:
            status = "EXPLORANDO"
            
        # Texto informativo
        marker.text = f"Confianza: {self.current_confidence:.3f}\nEstado: {status}\nPartículas: {self.particle_count}"
        
        # Tamaño del texto
        marker.scale.z = 0.12
        
        # Color según confianza
        if self.current_confidence > 0.7:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif self.current_confidence > 0.4:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        marker.color.a = 1.0
        
        # Tiempo de vida
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        
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