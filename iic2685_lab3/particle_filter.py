#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from scipy.spatial import KDTree
from random import gauss, uniform, randint

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64

class Particle:
    def __init__(self, x, y, ang, sigma=0.1):
        self.x, self.y, self.ang = x, y, ang
        self.last_x, self.last_y, self.last_ang = x, y, ang
        self.sigma = sigma
        self.weight = 1.0

    def move(self, dx, dy, dtheta):
        self.last_x, self.last_y, self.last_ang = self.x, self.y, self.ang
        self.x += dx + gauss(0, self.sigma)
        self.y += dy + gauss(0, self.sigma)
        self.ang += dtheta + gauss(0, self.sigma)

    def pos(self):
        return [self.x, self.y, self.ang]

    def last_pos(self):
        return [self.last_x, self.last_y, self.last_ang]

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parámetros del filtro
        self.num_particles = 1000
        self.sensor_noise = 0.1
        self.max_laser_range = 4.0

        # Estado del filtro
        self.particles = []
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.obstacle_coords = []

        # Estado del robot
        self.current_scan = None
        self.previous_pose = None
        self.localized = False

        # Suscriptores
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publicadores
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.best_pose_pub = self.create_publisher(PointStamped, '/best_pose', 10)
        self.confidence_pub = self.create_publisher(Float64, '/localization_confidence', 10)

        # Timer principal del filtro
        self.create_timer(0.1, self.particle_filter_step)
        
        self.get_logger().info("Filtro de partículas iniciado")

    def map_callback(self, msg):
        """Callback para recibir el mapa y inicializar partículas"""
        self.get_logger().info("Mapa recibido")
        
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_data = data

        self.obstacle_coords = [
            [self.map_origin[0] + x * self.map_resolution,
             self.map_origin[1] + y * self.map_resolution]
            for y in range(data.shape[0]) for x in range(data.shape[1])
            if data[y, x] > 50
        ]

        # Celdas libres
        free_cells = [
            [self.map_origin[0] + x * self.map_resolution,
             self.map_origin[1] + y * self.map_resolution]
            for y in range(data.shape[0]) for x in range(data.shape[1])
            if data[y, x] == 0
        ]

        # Inicializar partículas
        if free_cells:
            self.particles = [
                Particle(
                    *(np.array(free_cells[randint(0, len(free_cells)-1)]) +
                      np.random.uniform(-self.map_resolution, self.map_resolution, size=2).tolist()),
                    uniform(-math.pi, math.pi), sigma=0.1
                ) for _ in range(self.num_particles)
            ]
            self.get_logger().info(f"Inicializadas {len(self.particles)} partículas")

    def laser_callback(self, msg):
        """Callback LIDAR"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Callback odometría"""
        p = msg.pose.pose
        yaw = self.quaternion_to_yaw(p.orientation)
        
        if self.previous_pose and self.particles:
            # Calcular movimiento relativo
            dx = p.position.x - self.previous_pose.position.x
            dy = p.position.y - self.previous_pose.position.y
            dtheta = self.normalize_angle(yaw - self.quaternion_to_yaw(self.previous_pose.orientation))
            
            # Aplicar modelo de movimiento si hay movimiento significativo
            if any(abs(v) > 1e-3 for v in [dx, dy, dtheta]):
                for pt in self.particles:
                    pt.move(dx, dy, dtheta)
                    
        self.previous_pose = p

    def particle_filter_step(self):
        """Paso principal del filtro de partículas (MCL)"""
        if not (self.particles and self.current_scan and self.obstacle_coords):
            return

        # Paso de medición: actualizar pesos usando modelo de sensor
        self.measurement_update()
        
        # Verificar convergencia
        self.check_convergence()
        
        # Resampling (solo si no está localizado)
        if not self.localized:
            self.resample_particles()
        
        # Publicar resultados
        self.publish_particles()
        self.publish_best_pose()
        self.publish_confidence()

    def measurement_update(self):
        """Actualizar pesos de partículas usando modelo de sensor (Likelihood Fields)"""
        tree = KDTree(self.obstacle_coords)
        scan = self.current_scan
        total_w = 0.0

        for p in self.particles:
            likelihood = 1.0
            
            # Muestrear rayos para eficiencia (cada 20 rayos aproximadamente)
            for i in range(0, len(scan.ranges), max(1, len(scan.ranges)//20)):
                z = scan.ranges[i]
                
                # Ignorar lecturas inválidas
                if z >= self.max_laser_range or z <= scan.range_min:
                    continue
                    
                # Calcular punto final del rayo
                a = scan.angle_min + i * scan.angle_increment + p.ang
                x = p.x + z * math.cos(a)
                y = p.y + z * math.sin(a)
                
                # Encontrar distancia al obstáculo más cercano
                dist, _ = tree.query([x, y])
                
                # Calcular verosimilitud usando distribución gaussiana
                likelihood *= max(self.gaussian(dist, 0.0, self.sensor_noise), 1e-6)
                
            p.weight = likelihood
            total_w += likelihood

        # Normalizar pesos
        if total_w > 0:
            for p in self.particles:
                p.weight /= total_w

    def check_convergence(self):
        """Verificar si el filtro ha convergido (criterio de localización)"""
        if self.localized or not self.particles:
            return
            
        # Calcular estadísticas de las partículas
        positions = np.array([[p.x, p.y] for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        
        # Media ponderada de posiciones
        mean_pos = np.average(positions, axis=0, weights=weights)
        
        # Calcular dispersión espacial
        distances = np.sqrt(np.sum((positions - mean_pos)**2, axis=1))
        weighted_std = np.sqrt(np.average(distances**2, weights=weights))
        
        # Criterio de convergencia: dispersión < 0.5 metros
        if weighted_std < 0.5:
            self.localized = True
            self.get_logger().info(f"¡Robot localizado! Posición: ({mean_pos[0]:.3f}, {mean_pos[1]:.3f})")
            print(f"Pose localizada: x={mean_pos[0]:.3f}, y={mean_pos[1]:.3f}")

    def resample_particles(self):
        """Resampling sistemático de partículas"""
        weights = [p.weight for p in self.particles]
        
        # Evitar división por cero
        if sum(weights) == 0:
            return
            
        # Resampling con reemplazo usando numpy
        idx = np.random.choice(len(self.particles), self.num_particles, p=weights)
        
        # Crear nuevas partículas en las posiciones seleccionadas
        self.particles = [
            Particle(self.particles[i].x, self.particles[i].y, self.particles[i].ang, sigma=0.1)
            for i in idx
        ]

    def publish_particles(self):
        """Publicar partículas para visualización en RViz"""
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        for p in self.particles:
            pose = Pose()
            pose.position.x, pose.position.y = p.x, p.y
            pose.position.z = 0.0
            
            # Convertir ángulo a quaternion
            pose.orientation.w = math.cos(p.ang / 2)
            pose.orientation.z = math.sin(p.ang / 2)
            msg.poses.append(pose)
            
        self.particles_pub.publish(msg)

    def publish_best_pose(self):
        """Publicar mejor estimación de pose (media ponderada)"""
        if not self.particles:
            return
            
        weights = np.array([p.weight for p in self.particles])
        positions = np.array([[p.x, p.y] for p in self.particles])
        mean = np.average(positions, axis=0, weights=weights)
        
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x, msg.point.y = mean
        msg.point.z = 0.0
        
        self.best_pose_pub.publish(msg)

    def publish_confidence(self):
        """Publicar confianza de localización"""
        if not self.particles:
            return
            
        weights = np.array([p.weight for p in self.particles])
        max_w = np.max(weights)
        eff = 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0
        
        msg = Float64()
        msg.data = max_w * eff / self.num_particles
        self.confidence_pub.publish(msg)

    def quaternion_to_yaw(self, q):
        """Convertir quaternion a ángulo yaw"""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))

    def normalize_angle(self, a):
        """Normalizar ángulo al rango [-π, π]"""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def gaussian(self, x, mu, sigma):
        """Calcular probabilidad gaussiana"""
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

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