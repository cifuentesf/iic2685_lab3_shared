#!/usr/bin/env python3
"""
Actividad 2: Filtro de Partículas para Localización (Monte Carlo Localization)
Basado en Probabilistic Robotics, Cap. 8 (Thrun et al.)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Quaternion, Twist
from visualization_msgs.msg import MarkerArray, Marker
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from std_msgs.msg import Float32
import numpy as np
import yaml
import os
import cv2
from math import sin, cos, atan2, sqrt, pi, exp
from scipy.spatial import KDTree
from numpy.random import normal, uniform, choice
import threading

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        self.get_logger().info("Filtro de partículas para localización iniciado")

        # Parámetros del filtro
        self.num_particles = 500  # Número de partículas
        self.sigma_hit = 0.2      # Desviación estándar del modelo de sensor (metros)
        self.motion_noise = {
            'alpha1': 0.01,  # Error de rotación debido a rotación
            'alpha2': 0.01,  # Error de rotación debido a traslación
            'alpha3': 0.01,  # Error de traslación debido a traslación
            'alpha4': 0.01   # Error de traslación debido a rotación
        }
        
        # Cargar mapa
        self.map_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'mapas', 'mapa.yaml'
        )
        self.load_map()
        
        # Inicializar partículas
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialize_particles_uniform()
        
        # Variables de estado
        self.last_odom = None
        self.latest_scan = None
        self.initialized = False
        self.converged = False
        self.convergence_threshold = 0.5  # metros
        
        # Precalcular campo de verosimilitud
        self.precompute_likelihood_field()
        
        # Suscripciones
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        
        # Publicadores
        self.particles_pub = self.create_publisher(
            PoseArray, '/particle_cloud', 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, '/mcl_pose', 10
        )
        self.convergence_pub = self.create_publisher(
            Float32, '/particle_convergence', 10
        )
        
        # Timer para publicación periódica
        self.create_timer(0.1, self.publish_particles)
        self.create_timer(0.5, self.check_convergence)
        
        # Mutex para sincronización
        self.lock = threading.Lock()

    def load_map(self):
        """Carga el mapa y prepara estructuras de datos"""
        try:
            with open(self.map_path, 'r') as f:
                map_meta = yaml.safe_load(f)

            self.resolution = map_meta['resolution']
            self.origin = np.array(map_meta['origin'][:2])
            
            # Cargar imagen del mapa
            image_path = os.path.join(os.path.dirname(self.map_path), map_meta['image'])
            self.map_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.map_image = cv2.flip(self.map_image, 0)
            
            # Crear mapa binario: 0 = ocupado, 1 = libre
            self.binary_map = (self.map_image > 250).astype(np.uint8)
            self.height, self.width = self.binary_map.shape
            
            # Encontrar celdas libres para inicialización
            self.free_cells = np.column_stack(np.where(self.binary_map == 1))
            
            # KD-Tree de obstáculos
            obstacle_cells = np.column_stack(np.where(self.binary_map == 0))
            if len(obstacle_cells) > 0:
                self.obstacle_kdtree = KDTree(obstacle_cells)
            
            self.get_logger().info(
                f"Mapa cargado: {self.width}x{self.height}, "
                f"{len(self.free_cells)} celdas libres"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa: {str(e)}")

    def precompute_likelihood_field(self):
        """Precalcula el campo de verosimilitud para eficiencia"""
        self.get_logger().info("Precalculando campo de verosimilitud...")
        
        self.likelihood_field = np.zeros((self.height, self.width))
        sigma_pixels = self.sigma_hit / self.resolution
        
        # Para cada celda libre, calcular su verosimilitud
        for y in range(self.height):
            for x in range(self.width):
                if self.binary_map[y, x] == 1:  # Celda libre
                    dist, _ = self.obstacle_kdtree.query([y, x])
                    self.likelihood_field[y, x] = exp(-0.5 * (dist / sigma_pixels) ** 2)
        
        # Normalizar
        max_val = np.max(self.likelihood_field)
        if max_val > 0:
            self.likelihood_field /= max_val
            
        self.get_logger().info("Campo de verosimilitud calculado")

    def initialize_particles_uniform(self):
        """Inicializa partículas uniformemente en el espacio libre"""
        self.particles = []
        
        # Seleccionar celdas aleatorias
        if len(self.free_cells) > 0:
            indices = np.random.choice(len(self.free_cells), self.num_particles)
            
            for idx in indices:
                y, x = self.free_cells[idx]
                
                # Convertir a coordenadas del mundo
                wx = x * self.resolution + self.origin[0]
                wy = y * self.resolution + self.origin[1]
                wtheta = uniform(-pi, pi)
                
                particle = {
                    'x': wx,
                    'y': wy,
                    'theta': wtheta,
                    'weight': 1.0 / self.num_particles
                }
                self.particles.append(particle)
        
        self.get_logger().info(f"Inicializadas {self.num_particles} partículas")

    def odom_callback(self, msg):
        """Callback para odometría - aplica modelo de movimiento"""
        with self.lock:
            if self.last_odom is None:
                self.last_odom = msg
                return
            
            # Calcular movimiento relativo
            dx = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
            dy = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
            
            # Extraer orientaciones
            q1 = self.last_odom.pose.pose.orientation
            q2 = msg.pose.pose.orientation
            _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])
            _, _, yaw2 = euler_from_quaternion([q2.x, q2.y, q2.z, q2.w])
            
            # Calcular rotaciones
            delta_rot1 = atan2(dy, dx) - yaw1
            delta_trans = sqrt(dx*dx + dy*dy)
            delta_rot2 = yaw2 - yaw1 - delta_rot1
            
            # Normalizar ángulos
            delta_rot1 = self.normalize_angle(delta_rot1)
            delta_rot2 = self.normalize_angle(delta_rot2)
            
            # Aplicar modelo de movimiento a cada partícula
            if delta_trans > 0.01 or abs(delta_rot1) > 0.01 or abs(delta_rot2) > 0.01:
                self.apply_motion_model(delta_rot1, delta_trans, delta_rot2)
            
            self.last_odom = msg

    def apply_motion_model(self, drot1, dtrans, drot2):
        """
        Aplica el modelo de movimiento probabilístico (sample_motion_model_odometry)
        Algoritmo de Probabilistic Robotics, Table 5.6
        """
        for particle in self.particles:
            # Agregar ruido al movimiento
            drot1_hat = drot1 - self.sample_normal_distribution(
                self.motion_noise['alpha1'] * abs(drot1) + 
                self.motion_noise['alpha2'] * dtrans
            )
            
            dtrans_hat = dtrans - self.sample_normal_distribution(
                self.motion_noise['alpha3'] * dtrans + 
                self.motion_noise['alpha4'] * (abs(drot1) + abs(drot2))
            )
            
            drot2_hat = drot2 - self.sample_normal_distribution(
                self.motion_noise['alpha1'] * abs(drot2) + 
                self.motion_noise['alpha2'] * dtrans
            )
            
            # Actualizar pose de la partícula
            particle['x'] += dtrans_hat * cos(particle['theta'] + drot1_hat)
            particle['y'] += dtrans_hat * sin(particle['theta'] + drot1_hat)
            particle['theta'] += drot1_hat + drot2_hat
            particle['theta'] = self.normalize_angle(particle['theta'])

    def scan_callback(self, msg):
        """Callback para datos del LIDAR"""
        with self.lock:
            self.latest_scan = msg
            
            # Si tenemos scan, actualizar pesos y resamplear
            if len(self.particles) > 0:
                self.update_particle_weights(msg)
                self.resample_particles()
                self.estimate_pose()

    def update_particle_weights(self, scan):
        """
        Actualiza los pesos de las partículas basado en la observación
        Implementa el modelo de sensor Likelihood Fields
        """
        weights = []
        
        # Parámetros del scan
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        
        # Evaluar subconjunto de rayos para eficiencia
        step = max(1, len(scan.ranges) // 30)
        
        for particle in self.particles:
            weight = 1.0
            
            # Verificar si la partícula está en espacio libre
            mx, my = self.world_to_map(particle['x'], particle['y'])
            
            if (mx < 0 or mx >= self.width or my < 0 or my >= self.height or 
                self.binary_map[my, mx] == 0):
                weight = 1e-10  # Peso muy pequeño para partículas en obstáculos
            else:
                # Calcular verosimilitud basada en el scan
                for i in range(0, len(scan.ranges), step):
                    z = scan.ranges[i]
                    
                    if z >= scan.range_max or np.isnan(z) or np.isinf(z):
                        continue
                    
                    # Ángulo del rayo
                    angle = angle_min + i * angle_increment
                    
                    # Punto final del rayo
                    end_x = particle['x'] + z * cos(particle['theta'] + angle)
                    end_y = particle['y'] + z * sin(particle['theta'] + angle)
                    
                    # Convertir a coordenadas del mapa
                    end_mx, end_my = self.world_to_map(end_x, end_y)
                    
                    # Obtener verosimilitud del campo
                    if (0 <= end_mx < self.width and 0 <= end_my < self.height):
                        likelihood = self.likelihood_field[end_my, end_mx]
                        weight *= (likelihood + 0.1)  # Evitar ceros
            
            weights.append(weight)
        
        # Normalizar pesos
        weights = np.array(weights)
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            self.weights = weights / weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Actualizar pesos en partículas
        for i, particle in enumerate(self.particles):
            particle['weight'] = self.weights[i]

    def resample_particles(self):
        """
        Remuestreo de partículas usando Low Variance Resampling
        Algoritmo de Probabilistic Robotics, Table 4.4
        """
        new_particles = []
        
        # Low variance resampling
        M = self.num_particles
        r = uniform(0, 1.0/M)
        c = self.weights[0]
        i = 0
        
        for m in range(M):
            u = r + m * (1.0/M)
            while u > c and i < M-1:
                i += 1
                c += self.weights[i]
            
            # Crear nueva partícula (copiar la seleccionada)
            new_particle = {
                'x': self.particles[i]['x'],
                'y': self.particles[i]['y'],
                'theta': self.particles[i]['theta'],
                'weight': 1.0/M
            }
            
            # Agregar pequeño ruido para diversidad
            new_particle['x'] += normal(0, 0.01)
            new_particle['y'] += normal(0, 0.01)
            new_particle['theta'] += normal(0, 0.02)
            new_particle['theta'] = self.normalize_angle(new_particle['theta'])
            
            new_particles.append(new_particle)
        
        self.particles = new_particles
        self.weights = np.ones(M) / M

    def estimate_pose(self):
        """Estima la pose del robot basada en las partículas"""
        if len(self.particles) == 0:
            return
        
        # Calcular media ponderada
        x_mean = sum(p['x'] * p['weight'] for p in self.particles)
        y_mean = sum(p['y'] * p['weight'] for p in self.particles)
        
        # Para la orientación, usar vectores para evitar problemas de discontinuidad
        theta_x = sum(cos(p['theta']) * p['weight'] for p in self.particles)
        theta_y = sum(sin(p['theta']) * p['weight'] for p in self.particles)
        theta_mean = atan2(theta_y, theta_x)
        
        # Publicar pose estimada
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = x_mean
        pose_msg.pose.position.y = y_mean
        pose_msg.pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, theta_mean)
        pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        self.pose_pub.publish(pose_msg)
        
        # Log periódico
        if self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Cada segundo aprox
            self.get_logger().info(
                f"Pose estimada: x={x_mean:.3f}, y={y_mean:.3f}, θ={theta_mean:.3f}"
            )

    def check_convergence(self):
        """Verifica si el filtro ha convergido"""
        if len(self.particles) == 0:
            return
        
        # Calcular varianza de las partículas
        x_coords = [p['x'] for p in self.particles]
        y_coords = [p['y'] for p in self.particles]
        
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        # Métrica de convergencia (desviación estándar promedio)
        convergence = sqrt((x_var + y_var) / 2)
        
        # Publicar métrica
        conv_msg = Float32()
        conv_msg.data = convergence
        self.convergence_pub.publish(conv_msg)
        
        # Verificar si ha convergido
        if convergence < self.convergence_threshold and not self.converged:
            self.converged = True
            self.get_logger().info(
                f"¡FILTRO CONVERGIDO! Desviación: {convergence:.3f} m"
            )
            
            # Imprimir pose final
            x_mean = np.mean(x_coords)
            y_mean = np.mean(y_coords)
            self.get_logger().info(
                f"Localización completada en: x={x_mean:.3f}, y={y_mean:.3f}"
            )

    def publish_particles(self):
        """Publica las partículas para visualización en RViz"""
        if len(self.particles) == 0:
            return
        
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle['x']
            pose.position.y = particle['y']
            pose.position.z = 0.0
            
            q = quaternion_from_euler(0, 0, particle['theta'])
            pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            
            pose_array.poses.append(pose)
        
        self.particles_pub.publish(pose_array)

    def world_to_map(self, x, y):
        """Convierte coordenadas del mundo a índices del mapa"""
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)
        return mx, my

    def normalize_angle(self, angle):
        """Normaliza un ángulo al rango [-pi, pi]"""
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def sample_normal_distribution(self, variance):
        """Muestrea de una distribución normal con media 0"""
        return normal(0, sqrt(variance))

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()