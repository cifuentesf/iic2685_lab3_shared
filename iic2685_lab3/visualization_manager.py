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
        
        # Estado
        self.confidence = 0.0
        
        # Suscriptores
        self.create_subscription(PoseArray, '/particles', self.particles_callback, 10)
        self.create_subscription(PointStamped, '/best_pose', self.best_pose_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        # Publicadores
        self.particles_marker_pub = self.create_publisher(MarkerArray, '/particles_markers', 10)
        self.best_pose_marker_pub = self.create_publisher(Marker, '/best_pose_marker', 10)
        self.status_marker_pub = self.create_publisher(Marker, '/status_text', 10)
        
        self.get_logger().info("Gestor de visualización iniciado")
        
    def particles_callback(self, msg):
        marker_array = MarkerArray()
        
        # Limpiar markers anteriores
        clear_marker = Marker()
        clear_marker.header = msg.header
        clear_marker.ns = 'particles'
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Crear flecha para cada partícula
        for i, pose in enumerate(msg.poses):
            marker = Marker()
            marker.header = msg.header
            marker.ns = 'particles'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose
            
            # Tamaño
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            
            # Color según confianza
            marker.color.r = 1.0 - self.confidence
            marker.color.g = self.confidence
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 500000000
            
            marker_array.markers.append(marker)
            
        self.particles_marker_pub.publish(marker_array)
        
    def best_pose_callback(self, msg):
        """Visualizar mejor estimación"""
        # Esfera principal
        marker = Marker()
        marker.header = msg.header
        marker.ns = 'best_pose'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position = msg.point
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.9
        
        self.best_pose_marker_pub.publish(marker)
        
        # Cilindro vertical
        cylinder = Marker()
        cylinder.header = msg.header
        cylinder.ns = 'best_pose_cylinder'
        cylinder.id = 1
        cylinder.type = Marker.CYLINDER
        cylinder.action = Marker.ADD
        
        cylinder.pose.position = msg.point
        cylinder.pose.position.z = 0.5
        cylinder.pose.orientation.w = 1.0
        
        cylinder.scale.x = cylinder.scale.y = 0.05
        cylinder.scale.z = 1.0
        
        cylinder.color = marker.color
        
        self.best_pose_marker_pub.publish(cylinder)
        
    def confidence_callback(self, msg):
        self.confidence = msg.data
        
        marker = Marker()
        marker.header.frame_id = 'world_map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'status'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = -0.5
        marker.pose.position.y = 3.5
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        status = "LOCALIZADO" if self.confidence > 0.8 else "Localizando..."
        marker.text = f"Confianza: {self.confidence:.0%}\n{status}"
        
        marker.scale.z = 0.3
        
        # Color según confianza
        if self.confidence > 0.8:
            marker.color.r = 0.0
            marker.color.g = 1.0
        elif self.confidence > 0.5:
            marker.color.r = 1.0
            marker.color.g = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.status_marker_pub.publish(marker)
        

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()