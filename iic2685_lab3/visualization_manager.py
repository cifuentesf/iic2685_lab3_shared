#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseArray, PointStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray


class VisualizationManager(Node):
    def __init__(self):
        super().__init__('visualization_manager')
        
        # Estado interno
        self.current_confidence = 0.0
        self.particle_count = 0
        self.best_pose = None
        
        # Parámetros de visualización
        self.max_particles_display = 100  # Limitar para eficiencia
        self.particle_lifetime = 1.0  # segundos
        
        # Suscriptores
        self.create_subscription(PoseArray, '/particles', self.particles_callback, 10)
        self.create_subscription(PointStamped, '/best_pose', self.best_pose_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicadores
        self.particles_marker_pub = self.create_publisher(MarkerArray, '/particles_markers', 10)
        self.best_pose_marker_pub = self.create_publisher(Marker, '/best_pose_marker', 10)
        self.confidence_text_pub = self.create_publisher(Marker, '/confidence_text', 10)
        
        # Timer para actualización periódica
        self.create_timer(0.5, self.update_confidence_display)
        
        self.get_logger().info("Gestor de visualización iniciado")

    def particles_callback(self, msg):
        """Visualizar partículas como flechas en RViz"""
        marker_array = MarkerArray()
        
        # Marker para limpiar partículas anteriores
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = 'particles'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Limitar número de partículas mostradas para eficiencia
        num_particles = len(msg.poses)
        step = max(1, num_particles // self.max_particles_display)
        
        for i, pose in enumerate(msg.poses[::step]):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'particles'
            marker.id = i + 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Pose de la partícula
            marker.pose = pose
            
            # Tamaño de la flecha
            marker.scale.x = 0.08  # Longitud
            marker.scale.y = 0.02  # Ancho
            marker.scale.z = 0.02  # Alto
            
            # Color azul con transparencia
            marker.color.r = 0.0
            marker.color.g = 0.3
            marker.color.b = 1.0
            marker.color.a = 0.6
            
            # Tiempo de vida
            marker.lifetime = rclpy.duration.Duration(seconds=self.particle_lifetime).to_msg()
            
            marker_array.markers.append(marker)
        
        self.particle_count = num_particles
        self.particles_marker_pub.publish(marker_array)

    def best_pose_callback(self, msg):
        """Visualizar mejor estimación de pose"""
        self.best_pose = msg
        
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
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        # Color verde brillante
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        
        # Tiempo de vida
        marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
        
        self.best_pose_marker_pub.publish(marker)

    def confidence_callback(self, msg):
        """Actualizar confianza de localización"""
        self.current_confidence = msg.data

    def update_confidence_display(self):
        """Actualizar display de confianza y estado"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'confidence_info'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Posición del texto (esquina superior del mapa)
        marker.pose.position.x = 0.5
        marker.pose.position.y = 2.5
        marker.pose.position.z = 0.8
        marker.pose.orientation.w = 1.0
        
        # Determinar estado basado en confianza
        if self.current_confidence >= 0.8:
            status = "LOCALIZADO ✓"
            color = (0.0, 1.0, 0.0)  # Verde
        elif self.current_confidence >= 0.5:
            status = "CONVERGIENDO..."
            color = (1.0, 1.0, 0.0)  # Amarillo
        elif self.current_confidence >= 0.2:
            status = "EXPLORANDO"
            color = (1.0, 0.7, 0.0)  # Naranja
        else:
            status = "INICIALIZANDO"
            color = (1.0, 0.0, 0.0)  # Rojo
        
        # Texto informativo
        marker.text = f"Confianza: {self.current_confidence:.3f}\n"
        marker.text += f"Estado: {status}\n"
        marker.text += f"Partículas: {self.particle_count}"
        
        # Estilo del texto
        marker.scale.z = 0.12  # Tamaño del texto
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        
        # Tiempo de vida
        marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
        
        self.confidence_text_pub.publish(marker)
        
        # Log ocasional para debugging
        if self.particle_count > 0:
            self.get_logger().info(
                f"Confianza: {self.current_confidence:.3f}, "
                f"Partículas: {self.particle_count}, "
                f"Estado: {status}",
                throttle_duration_sec=5.0
            )


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