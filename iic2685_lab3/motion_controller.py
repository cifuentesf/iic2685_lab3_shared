#!/usr/bin/env python3
# Actividad 3 - Exploración Reactiva + Evaluación del Filtro de Partículas

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np

class ReactiveExplorer(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.get_logger().info("Nodo de exploración reactiva iniciado")

        # Parámetros de control
        self.move_time = 2.5         # segundos de avance
        self.turn_time = 1.5         # segundos de giro
        self.state = "move"          # estado actual
        self.state_start_time = self.get_clock().now()
        self.safe_distance = 0.5     # distancia de seguridad frontal (m)

        # Control de movimiento
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_mux/input/navigation', 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # Timer principal
        self.timer = self.create_timer(0.1, self.control_loop)

        # Variables de sensor
        self.min_front = 999.0

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        front = ranges[len(ranges)//3 : 2*len(ranges)//3]  # parte frontal
        self.min_front = np.nanmin(front)

    def control_loop(self):
        now = self.get_clock().now()
        elapsed = (now - self.state_start_time).nanoseconds / 1e9

        twist = Twist()

        if self.state == "move":
            if self.min_front < self.safe_distance:
                self.get_logger().info("Obstáculo detectado. Cambiando a giro.")
                self.state = "turn"
                self.state_start_time = now
                return

            twist.linear.x = 0.2
            twist.angular.z = 0.0

            if elapsed > self.move_time:
                self.get_logger().info("Moviendo → Giro")
                self.state = "turn"
                self.state_start_time = now

        elif self.state == "turn":
            twist.linear.x = 0.0
            twist.angular.z = 0.5

            if elapsed > self.turn_time:
                self.get_logger().info("Girando → Movimiento")
                self.state = "move"
                self.state_start_time = now

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
