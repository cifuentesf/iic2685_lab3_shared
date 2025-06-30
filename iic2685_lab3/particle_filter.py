#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from random import uniform, gauss, choices
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from std_msgs.msg import Float64, Header
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion


class Particle:
    def __init__(self, x, y, ang, sigma=0.02):
        self.x, self.y, self.ang = x, y, ang
        self.weight = 1.0
        self.sigma = sigma

    def move(self, delta_x, delta_y, delta_ang):
        self.x += delta_x + gauss(0, self.sigma)
        self.y += delta_y + gauss(0, self.sigma)
        self.ang += delta_ang + gauss(0, self.sigma * 0.5)
        self.ang = self.normalize_angle(self.ang)

    def pos(self):
        return [self.x, self.y, self.ang]
    
    @staticmethod
    def normalize_angle(angle):
        """Normalizar ángulo"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parámetros del filtro MCL
        self.num_particles = 200
        self.sigma_motion = 0.03
        self.sigma_hit = 0.15
        self.z_hit = 0.95
        self.z_random = 0.05
        self.z_max = 3.5
        self.min_distance_from_obstacles = 0.3  # Distancia mínima a obstáculos para inicialización
        
        # Estado del filtro
        self.particles = []
        self.map_data = None
        self.map_info = None
        self.current_scan = None
        self.last_odom = None
        self.prev_odom = None
        
        # Campo de verosimilitud precalculado
        self.likelihood_field = None
        
        # Cargar mapa y inicializar
        self.load_map()
        self.precompute_likelihood_field()
        self.initialize_particles()
        
        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publicadores
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)
        
        # Timer para MCL
        self.create_timer(0.2, self.mcl_update)
        
        self.get_logger().info(f"Filtro MCL iniciado con {self.num_particles} partículas")

    def load_map(self):
        try:
            import os
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('iic2685_lab3')
            map_path = os.path.join(pkg_share, 'maps', 'mapa.pgm')
            self.map_data = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            self.map_info = {
                'resolution': 0.01,
                'origin': [0.0, 0.0, 0.0],
                'width': self.map_data.shape[1],
                'height': self.map_data.shape[0]
            }
            self.get_logger().info(f"Mapa cargado: {self.map_info['width']}x{self.map_info['height']}")
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {e}")
            self.map_data = np.ones((270, 270), dtype=np.uint8) * 255
            self.map_info = {'resolution': 0.01, 'origin': [0.0, 0.0, 0.0], 'width': 270, 'height': 270}

    def precompute_likelihood_field(self):
        obstacle_pixels = np.where(self.map_data < 100)
        if len(obstacle_pixels[0]) == 0:
            self.likelihood_field = np.ones_like(self.map_data, dtype=np.float32) * 0.1
            return
        h, w = self.map_data.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        all_coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        obstacle_coords = np.column_stack([obstacle_pixels[0], obstacle_pixels[1]])
        distances = cdist(all_coords, obstacle_coords, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        min_distances_meters = min_distances * self.map_info['resolution']
        likelihood_values = np.exp(-0.5 * (min_distances_meters / self.sigma_hit) ** 2)
        likelihood_values = likelihood_values / np.max(likelihood_values)
        self.likelihood_field = likelihood_values.reshape(h, w).astype(np.float32)

    def initialize_particles(self):
        self.particles = []
        free_space_mask = self.map_data > 200
        erosion_pixels = int(self.min_distance_from_obstacles / self.map_info['resolution'])
        kernel = np.ones((erosion_pixels*2+1, erosion_pixels*2+1), np.uint8)
        eroded_mask = binary_erosion(free_space_mask, structure=kernel)
        safe_cells = np.where(eroded_mask)
        if len(safe_cells[0]) == 0:
            safe_cells = np.where(free_space_mask)
            if len(safe_cells[0]) == 0:
                for _ in range(self.num_particles):
                    x = uniform(0.5, (self.map_info['width'] - 1) * self.map_info['resolution'])
                    y = uniform(0.5, (self.map_info['height'] - 1) * self.map_info['resolution'])
                    ang = uniform(-np.pi, np.pi)
                    self.particles.append(Particle(x, y, ang, self.sigma_motion))
                return
        attempts = 0
        max_attempts = 10000
        while len(self.particles) < self.num_particles and attempts < max_attempts:
            idx = np.random.randint(len(safe_cells[0]))
            py, px = safe_cells[0][idx], safe_cells[1][idx]
            x = px * self.map_info['resolution'] + self.map_info['origin'][0]
            y = py * self.map_info['resolution'] + self.map_info['origin'][1]
            ang = uniform(-np.pi, np.pi)
            margin = 0.1
            if (margin <= x <= (self.map_info['width'] * self.map_info['resolution'] - margin) and
                margin <= y <= (self.map_info['height'] * self.map_info['resolution'] - margin)):
                self.particles.append(Particle(x, y, ang, self.sigma_motion))
            attempts += 1
        for particle in self.particles:
            particle.weight = 1.0 / self.num_particles

    def laser_callback(self, msg):
        self.current_scan = msg

    def odom_callback(self, msg):
        self.last_odom = msg

    def mcl_update(self):
        if self.current_scan is None:
            return
        if self.last_odom is not None and self.prev_odom is not None:
            scan_time = self.current_scan.header.stamp.sec + self.current_scan.header.stamp.nanosec * 1e-9
            odom_time = self.last_odom.header.stamp.sec + self.last_odom.header.stamp.nanosec * 1e-9
            time_diff = abs(scan_time - odom_time)
            if time_diff < 0.1:
                self.sample_motion_model()
        self.measurement_model_update()
        self.normalize_weights()
        ess = self.effective_sample_size()
        if ess < self.num_particles * 0.5:
            self.resample()
        confidence = self.calculate_confidence()
        self.publish_particles()
        self.publish_best_pose()
        self.publish_confidence(confidence)

    def sample_motion_model(self):
        if self.prev_odom is None:
            self.prev_odom = self.last_odom
            return
        try:
            dx = self.last_odom.pose.pose.position.x - self.prev_odom.pose.pose.position.x
            dy = self.last_odom.pose.pose.position.y - self.prev_odom.pose.pose.position.y
            curr_yaw = euler_from_quaternion([
                self.last_odom.pose.pose.orientation.x,
                self.last_odom.pose.pose.orientation.y,
                self.last_odom.pose.pose.orientation.z,
                self.last_odom.pose.pose.orientation.w
            ])[2]
            prev_yaw = euler_from_quaternion([
                self.prev_odom.pose.pose.orientation.x,
                self.prev_odom.pose.pose.orientation.y,
                self.prev_odom.pose.pose.orientation.z,
                self.prev_odom.pose.pose.orientation.w
            ])[2]
            dyaw = Particle.normalize_angle(curr_yaw - prev_yaw)
            movement_magnitude = np.sqrt(dx*dx + dy*dy)
            noise_scale = max(0.001, movement_magnitude * 0.1)
            for particle in self.particles:
                particle.move(dx, dy, dyaw)
                particle.x += np.random.normal(0, noise_scale)
                particle.y += np.random.normal(0, noise_scale)
                particle.ang += np.random.normal(0, noise_scale * 0.1)
                particle.ang = Particle.normalize_angle(particle.ang)
            self.prev_odom = self.last_odom
        except Exception as e:
            self.get_logger().error(f"Error en modelo de movimiento: {e}")

    def measurement_model_update(self):
        if self.current_scan is None:
            return
        ranges = np.array(self.current_scan.ranges)
        ranges[~np.isfinite(ranges)] = self.z_max
        ranges[ranges <= 0.0] = self.z_max
        ranges = np.clip(ranges, 0.1, self.z_max)
        angle_min = self.current_scan.angle_min
        angle_inc = self.current_scan.angle_increment
        step = max(1, len(ranges) // 20)
        for particle in self.particles:
            likelihood = 1.0
            for i in range(0, len(ranges), step):
                z_measured = ranges[i]
                if z_measured >= self.z_max:
                    continue
                beam_angle = angle_min + i * angle_inc + particle.ang
                x_z = particle.x + z_measured * np.cos(beam_angle)
                y_z = particle.y + z_measured * np.sin(beam_angle)
                px = int((x_z - self.map_info['origin'][0]) / self.map_info['resolution'])
                py = int((y_z - self.map_info['origin'][1]) / self.map_info['resolution'])
                if 0 <= px < self.map_info['width'] and 0 <= py < self.map_info['height']:
                    p_hit = self.z_hit * self.likelihood_field[py, px]
                else:
                    p_hit = 0.01
                p_random = self.z_random / self.z_max
                beam_likelihood = p_hit + p_random
                likelihood *= beam_likelihood
            particle.weight = likelihood

    def normalize_weights(self):
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0:
            for particle in self.particles:
                particle.weight = 1.0 / self.num_particles
        else:
            weights_sum = np.sum(weights)
            for i, particle in enumerate(self.particles):
                particle.weight = weights[i] / weights_sum

    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0:
            weights = np.ones(len(weights))
        weights /= np.sum(weights)
        indices = choices(range(len(self.particles)), weights=weights, k=self.num_particles)
        new_particles = []
        for i in indices:
            old_particle = self.particles[i]
            new_particle = Particle(old_particle.x, old_particle.y, old_particle.ang, self.sigma_motion)
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        self.particles = new_particles

    def effective_sample_size(self):
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0:
            return self.num_particles
        weights /= np.sum(weights)
        return 1.0 / np.sum(weights ** 2)

    def calculate_confidence(self):
        if len(self.particles) == 0:
            return 0.0
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0:
            return 0.0
        weights /= np.sum(weights)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(self.particles))
        confidence = 1.0 - (entropy / max_entropy)
        return confidence

    def publish_particles(self):
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.frame_id = "odom"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            pose.position.z = 0.0
            quat = quaternion_from_euler(0, 0, particle.ang)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            pose_array.poses.append(pose)
        self.particles_pub.publish(pose_array)

    def publish_best_pose(self):
        if not self.particles:
            return
        weights = [p.weight for p in self.particles]
        best_idx = np.argmax(weights)
        best_particle = self.particles[best_idx]
        point_msg = PointStamped()
        point_msg.header.frame_id = "odom"
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = best_particle.x
        point_msg.point.y = best_particle.y
        point_msg.point.z = 0.0
        self.best_pose_pub.publish(point_msg)

    def publish_confidence(self, confidence):
        confidence_msg = Float64()
        confidence_msg.data = confidence
        self.confidence_pub.publish(confidence_msg)
def main(args=None):
    rclpy.init(args=args)
    
    particle_filter = ParticleFilter()
    
    try:
        rclpy.spin(particle_filter)
    except KeyboardInterrupt:
        pass
    finally:
        particle_filter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()