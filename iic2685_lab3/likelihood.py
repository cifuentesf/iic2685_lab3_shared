#!/usr/bin/env python3
from scipy import spatial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32 
import cv2
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import Vector3, Pose
from tf_transformations import euler_from_quaternion

class likelihood(Node):
    def __init__(self):
        super().__init__('likelihood')
        self.sigma = 10.0  # Desviación estándar para el campo de probabilidad
        self.z_hit = 1.0  # Ponderador de ajuste de probabilidad
        self.matriz = self.calcular_likelihood()
        self.lidar_cb_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.pose_sub = self.create_subscription(Pose, '/real_pose', self.pose_cb, 1 )
        self.q_pub = self.create_publisher(Float32, '/likelihood', 10)

        self.q_list = []

        self.resolution = 0.01  # Resolución del mapa 0.01 m/píxel



    def lidar_cb(self, data):
        self.get_logger().info("Lidar callback triggered")

        # Se simulan los 57° del Kinekt
        lidar = data.ranges[61:118]  # 57° = 118-61
        #self.get_logger().info(f"Lidar data received: {lidar}, length: {len(lidar)} ")
        q = 1
        for i in range(len(lidar)):
            # self.get_logger().info(f"Processing range {i}: {lidar[i]}")
            if lidar[i] < 4.0:
                #x = self.robot_pose[0] + lidar[i] * np.cos((self.robot_pose[2])*np.pi/180)  # 0.0174533 rad = 1° en radianes
                #y = self.robot_pose[1] + lidar[i] * np.sin((self.robot_pose[2])*np.pi/180)
                #x = self.robot_pose[0] + lidar[i] * np.cos((self.robot_pose[2]+ (i-29))*np.pi/180)  
                #y = self.robot_pose[1] + lidar[i] * np.sin((self.robot_pose[2]+ (i-29))*np.pi/180)
                x = 0.5 + lidar[i] * np.cos((0.0 + (i-29))*np.pi/180) 
                y = 0.5 + lidar[i] * np.sin((0.0 + (i-29))*np.pi/180)
                x_pix = int(x / self.resolution ) 
                y_pix = int(y / self.resolution ) 
                #self.get_logger().info(f"P Coords: ({x_pix}, {y_pix})")
                #self.get_logger().info(f"img cords{270-y_pix}, {x_pix}")

                if 0 <= x_pix < self.matriz.shape[1] and 0 <= 270-y_pix < self.matriz.shape[0]: # Matriz de 270x270
                    q *= self.matriz[270-y_pix, x_pix]*self.z_hit
                    #self.get_logger().info(f"Likelihood value: {self.matriz[270-y_pix, x_pix]}")
                    self.get_logger().info(f"Updated likelihood value: {q}")
                else:
                    self.get_logger().warn(f"Pixel coordinates out of bounds: ({x_pix}, {y_pix})")
        self.q_list.append( q )
        q_msg = Float32()
        q_msg.data = q
        self.q_pub.publish(q_msg)
        #self.get_logger().info(f"Likelihood value for this scan: {q}")
        #self.get_logger().info(f"Current likelihood list: {self.q_list}")
                
        
    def pose_cb(self, pose):
        x = pose.position.x
        y = pose.position.y
        roll, pitch, yaw = euler_from_quaternion( ( pose.orientation.x,
                                                    pose.orientation.y,
                                                    pose.orientation.z,
                                                    pose.orientation.w ) )
        
        self.robot_pose = [x, y, yaw]
        #self.get_logger().info(f"Pose updated: x={x}, y={y}, yaw={yaw}")

    def calcular_likelihood(self):
        self.get_logger().info("Calculating likelihood field...")

        mapa_path = '/root/mp_ws/src/lab3/nodes/data/mapa.pgm'
        map_img = cv2.imread(mapa_path, cv2.IMREAD_GRAYSCALE)
        
        self.get_logger().info(f"Mapa cargado con forma: {map_img.shape}")

        # Binarizar el mapa: obstáculos = 1, libre = 0 
        obstacle_map = (map_img < 200).astype(np.uint8)
        #self.get_logger().info(f"Mapa de obstáculos binarizado con forma: {obstacle_map[:20, :20]}...")  # Muestra una pequeña parte del mapa

        # Calcular el campo de distancias (Distance Transform) ---
        dist_map = cv2.distanceTransform(1 - obstacle_map, cv2.DIST_L2, 5)
        #self.get_logger().info(f"Mapa de distancias calculado con forma: {dist_map[:20, :20]}")

        # Crear el campo de probabilidad (Likelihood Field)
        sigma = self.sigma # ajusta según resolución del mapa (en píxeles)
        prob_map = np.exp(- (dist_map**2) / (2 * sigma**2))
        #self.get_logger().info(f"Mapa de probabilidad creado con forma: {prob_map[:20, :20]}")

        # Normalizar imágenes para visualización
        dist_vis = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        prob_vis = cv2.normalize(prob_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Dar colores epicos (calor)
        dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_HOT)
        prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_HOT)
        map_color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)

        # --- 6. Mostrar imágenes ---
        # cv2.imshow("Mapa original", map_color)
        # cv2.imshow("Campo de distancias", dist_vis)
        cv2.imshow("Likelihood field (Probabilidad)", prob_color)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return prob_map

def main(args=None):
    rclpy.init(args=args)
    likelihood_node = likelihood()
    rclpy.spin(likelihood_node)
    likelihood_node.destroy_node()
    rclpy.shutdown()

    

if __name__ == '__main__':
    main()