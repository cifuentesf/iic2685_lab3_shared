#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        
        self.step_distance = 0.15
        self.linear_speed = 0.12
        self.angular_speed = 0.3
        self.confidence_threshold = 0.8
        
        self.desired_wall_distance = 0.4
        self.collision_distance = 0.35
        self.wall_following_active = True
        
        self.kp = 1.2
        self.ki = 0.01
        self.kd = 0.08
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_time = None
        
        self.state = "filtering"
        self.current_scan = None
        self.localization_confidence = 0.0
        self.filter_iterations = 0
        self.max_filter_iterations = 25
        self.step_count = 0
        
        self.movement_start_time = None
        self.movement_duration = 0.0
        self.movement_type = "forward"
        self.is_moving = False
        
        self.left_distance = float('inf')
        self.left_front_distance = float('inf')
        self.front_distance = float('inf')
        self.right_front_distance = float('inf')
        self.right_distance = float('inf')
        
        self.left_wall_distance = float('inf')
        self.right_wall_distance = float('inf')
        
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Float64, '/localization_confidence', self.confidence_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navegador mejorado iniciado - Fase de filtrado inicial")

    def laser_callback(self, msg):
        self.current_scan = msg
        self.process_scan_data(msg)

    def process_scan_data(self, scan):
        if scan is None:
            return
            
        ranges = np.array(scan.ranges)
        ranges[~np.isfinite(ranges)] = 4.0
        ranges[ranges <= 0.0] = 4.0
        ranges[ranges > 3.9] = 4.0
        
        n = len(ranges)
        if n == 0:
            return
            
        sector_size = n // 5
        
        right_sector = ranges[0:sector_size]
        self.right_distance = np.percentile(right_sector, 25) if len(right_sector) > 0 else float('inf')
        
        right_front_sector = ranges[sector_size:2*sector_size]
        self.right_front_distance = np.percentile(right_front_sector, 25) if len(right_front_sector) > 0 else float('inf')
        
        front_sector = ranges[2*sector_size:3*sector_size]
        self.front_distance = np.percentile(front_sector, 15) if len(front_sector) > 0 else float('inf')
        
        left_front_sector = ranges[3*sector_size:4*sector_size]
        self.left_front_distance = np.percentile(left_front_sector, 25) if len(left_front_sector) > 0 else float('inf')
        
        left_sector = ranges[4*sector_size:]
        self.left_distance = np.percentile(left_sector, 25) if len(left_sector) > 0 else float('inf')
        
        self.left_wall_distance = self.left_distance
        self.right_wall_distance = self.right_distance

    def confidence_callback(self, msg):
        self.localization_confidence = msg.data
        
        if msg.data >= self.confidence_threshold and self.state != "localized":
            self.state = "localized"
            self.get_logger().info(f"¡Robot localizado! Confianza: {msg.data:.3f}")
            self.get_logger().info(f"Localización completada en {self.step_count} pasos")
            self.create_timer(2.0, self.start_post_localization_navigation, one_shot=True)

    def start_post_localization_navigation(self):
        self.state = "exploring"
        self.get_logger().info("Iniciando exploración continua post-localización")

    def control_loop(self):
        if self.state == "filtering":
            self.filtering_phase()
        elif self.state == "moving":
            self.moving_phase()
        elif self.state == "localized":
            self.stop_robot()
        elif self.state == "exploring":
            self.continuous_navigation()

    def filtering_phase(self):
        self.stop_robot()
        self.filter_iterations += 1
        
        if self.filter_iterations >= self.max_filter_iterations:
            self.get_logger().info(f"Completadas {self.filter_iterations} iteraciones de filtro")
            self.get_logger().info(f"Iniciando movimiento #{self.step_count + 1}")
            
            self.filter_iterations = 0
            self.state = "moving"
            self.determine_movement()
            self.execute_movement()

    def determine_movement(self):
        if self.current_scan is None:
            self.movement_type = "forward"
            return
            
        frontal_obstruction = (self.front_distance < self.collision_distance or 
                             self.left_front_distance < self.collision_distance or 
                             self.right_front_distance < self.collision_distance)
        
        if frontal_obstruction:
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            
            if left_space > right_space + 0.15:
                self.movement_type = "turn_left"
            elif right_space > left_space + 0.15:
                self.movement_type = "turn_right"
            else:
                if self.left_distance < 1.2:
                    self.movement_type = "turn_right"
                else:
                    self.movement_type = "turn_left"
        else:
            if self.left_distance < 1.0:
                wall_error = self.left_distance - self.desired_wall_distance
                
                if abs(wall_error) < 0.08:
                    self.movement_type = "forward"
                elif wall_error > 0.15:
                    self.movement_type = "turn_left"
                elif wall_error < -0.15:
                    self.movement_type = "turn_right"
                else:
                    self.movement_type = "forward"
            else:
                if self.right_distance < self.left_distance:
                    self.movement_type = "turn_right"
                else:
                    self.movement_type = "turn_left"
        
        self.get_logger().info(
            f"Mov: {self.movement_type} | "
            f"L:{self.left_distance:.2f} LF:{self.left_front_distance:.2f} "
            f"F:{self.front_distance:.2f} RF:{self.right_front_distance:.2f} "
            f"R:{self.right_distance:.2f}"
        )

    def execute_movement(self):
        self.movement_start_time = self.get_clock().now()
        self.is_moving = True
        
        if self.movement_type == "forward":
            self.movement_duration = self.step_distance / self.linear_speed
        else:
            rotation_angle = np.pi/3
            self.movement_duration = rotation_angle / self.angular_speed
        
        self.send_movement_command()

    def send_movement_command(self):
        cmd = Twist()
        
        if self.movement_type == "forward":
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.0
        elif self.movement_type == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        elif self.movement_type == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = -self.angular_speed
        
        self.cmd_pub.publish(cmd)

    def moving_phase(self):
        if not self.is_moving:
            return
            
        current_time = self.get_clock().now()
        elapsed = (current_time - self.movement_start_time).nanoseconds * 1e-9
        
        if elapsed < self.movement_duration:
            self.send_movement_command()
        else:
            self.stop_robot()
            self.is_moving = False
            self.step_count += 1
            
            self.state = "filtering"
            self.get_logger().info(f"Movimiento completado. Volviendo a filtrado (paso {self.step_count})")

    def continuous_navigation(self):
        if self.current_scan is None:
            return
            
        cmd = Twist()
        
        frontal_clear = (self.front_distance > self.collision_distance and 
                        self.left_front_distance > self.collision_distance and 
                        self.right_front_distance > self.collision_distance)
        
        if not frontal_clear:
            left_space = min(self.left_distance, self.left_front_distance)
            right_space = min(self.right_distance, self.right_front_distance)
            
            if left_space > right_space + 0.1:
                cmd.angular.z = 0.5
                cmd.linear.x = 0.05
            elif right_space > left_space + 0.1:
                cmd.angular.z = -0.5
                cmd.linear.x = 0.05
            else:
                cmd.angular.z = 0.6
                cmd.linear.x = 0.0
                
        elif self.left_distance < 1.2:
            current_time = self.get_clock().now()
            
            if self.last_time is not None:
                dt = (current_time - self.last_time).nanoseconds * 1e-9
                
                if dt > 0:
                    wall_error = self.left_distance - self.desired_wall_distance
                    
                    if self.left_front_distance < self.desired_wall_distance * 0.8:
                        wall_error -= 0.1
                    
                    proportional = self.kp * wall_error
                    self.integral_error += wall_error * dt
                    self.integral_error = np.clip(self.integral_error, -0.5, 0.5)
                    integral = self.ki * self.integral_error
                    derivative = self.kd * (wall_error - self.previous_error) / dt
                    
                    angular_output = -(proportional + integral + derivative)
                    cmd.angular.z = np.clip(angular_output, -0.7, 0.7)
                    
                    if abs(wall_error) < 0.1:
                        cmd.linear.x = self.linear_speed
                    elif abs(wall_error) < 0.2:
                        cmd.linear.x = self.linear_speed * 0.8
                    else:
                        cmd.linear.x = self.linear_speed * 0.6
                    
                    self.previous_error = wall_error
            
            self.last_time = current_time
            
        else:
            if self.right_distance < self.left_distance and self.right_distance < 1.5:
                cmd.linear.x = self.linear_speed * 0.8
                cmd.angular.z = -0.15
            else:
                cmd.linear.x = self.linear_speed * 0.9
                cmd.angular.z = 0.2
        
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()