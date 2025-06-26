#!/usr/bin/env python3
# Actividad 2 - Filtro de Partículas (Monte Carlo Localization)

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Quaternion
from tf_transformations import quaternion_from_euler
import numpy as np
import yaml
import os
import cv2
from math import sin, cos, atan2, sqrt, pi
from scipy.spatial import KDTree
from random import choices

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        self.get_logger().info("Filtro de partículas iniciado")

        # Parámetros configurables
        self.N = 100  # número de partículas
        self.sigma_sensor = 3.0  # desviación gaussiana en píxeles
        self.sigma_xy = 0.02  # ruido en movimiento (m)
        self.sigma_theta = 0.05  # ruido angular (rad)

        # Mapa
        self.map_path = os.path.join(os.path.dirname(__file__), '..', 'mapas', 'mapa.yaml')
        self.load_map()

        # Partículas
        self.particles = self.initialize_particles()

        # Últimas mediciones
        self.last_odom = None
        self.latest_scan = None

        # KDTree para distancias
        self.kdtree = KDTree(np.column_stack(np.where(self.binary_map == 1)))

        # Subscripciones
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publicadores
        self.particle_pub = self.create_publisher(PoseArray, '/pf/particles', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/pose', 10)

    def load_map(self):
        with open(self.map_path, 'r') as f:
            map_meta = yaml.safe_load(f)

        self.resolution = map_meta['resolution']
        self.origin = map_meta['origin'][:2]

        image_path = os.path.join(os.path.dirname(self.map_path), map_meta['image'])
        occ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        occ = cv2.flip(occ, 0)
        self.binary_map = np.where(occ < 250, 1, 0)
        self.height, self.width = self.binary_map.shape

    def initialize_particles(self):
        '''Inicializa partículas uniformemente sobre zonas libres del mapa'''
        free_indices = np.column_stack(np.where(self.binary_map == 0))
        indices = free_indices[np.random.choice(free_indices.shape[0], self.N)]
        particles = []
        for y, x in indices:
            wx = x * self.resolution + self.origin[0]
            wy = y * self.resolution + self.origin[1]
            theta = np.random.uniform(-pi, pi)
            particles.append([wx, wy, theta])
        return np.array(particles)

    def odom_callback(self, msg):
        if self.last_odom is None:
            self.last_odom = msg
            return

        dx = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
        dist = sqrt(dx**2 + dy**2)

        # cambio angular
        q1 = self.last_odom.pose.pose.orientation
        q2 = msg.pose.pose.orientation
        yaw1 = 2 * atan2(q1.z, q1.w)
        yaw2 = 2 * atan2(q2.z, q2.w)
        dtheta = yaw2 - yaw1

        # Modelo de movimiento probabilístico
        for i in range(self.N):
            x, y, theta = self.particles[i]
            noisy_dist = dist + np.random.normal(0, self.sigma_xy)
            noisy_dtheta = dtheta + np.random.normal(0, self.sigma_theta)
            theta += noisy_dtheta
            x += noisy_dist * cos(theta)
            y += noisy_dist * sin(theta)
            self.particles[i] = [x, y, theta]

        self.last_odom = msg
        self.try_update_filter()

    def scan_callback(self, msg):
        self.latest_scan = msg

    def try_update_filter(self):
        '''Solo actualiza si hay escaneo disponible'''
        if self.latest_scan is None:
            return

        weights = np.zeros(self.N)

        angles = np.arange(
            self.latest_scan.angle_min,
            self.latest_scan.angle_max,
            self.latest_scan.angle_increment
        )

        max_range = self.latest_scan.range_max

        for i, (x, y, theta) in enumerate(self.particles):
            prob = 0.0
            for j, r in enumerate(self.latest_scan.ranges[::5]):  # menos rayos para velocidad
                if r >= max_range or np.isnan(r):
                    continue

                angle = angles[j*5]
                zx = x + r * cos(theta + angle)
                zy = y + r * sin(theta + angle)

                mx = int((zx - self.origin[0]) / self.resolution)
                my = int((zy - self.origin[1]) / self.resolution)

                if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
                    continue

                dist_pix, _ = self.kdtree.query([my, mx])
                prob += np.exp(-0.5 * (dist_pix / self.sigma_sensor) ** 2)

            weights[i] = prob + 1e-6  # evitar ceros

        # Normalizar
        weights /= np.sum(weights)

        # Resampling
        indices = choices(range(self.N), weights=weights, k=self.N)
        self.particles = self.particles[indices]

        # Publicar estimado
        self.publish_estimate()
        self.publish_particles()

    def publish_particles(self):
        msg = PoseArray()
        msg.header.frame_id = "map"
        for x, y, theta in self.particles:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            quat = quaternion_from_euler(0, 0, theta)
            pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            msg.poses.append(pose)
        self.particle_pub.publish(msg)

    def publish_estimate(self):
        mean_pose = np.mean(self.particles, axis=0)
        x, y, theta = mean_pose
        self.get_logger().info(f"Estimado: x={x:.2f}, y={y:.2f}, θ={theta:.2f}")

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        quat = quaternion_from_euler(0, 0, theta)
        pose_msg.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
