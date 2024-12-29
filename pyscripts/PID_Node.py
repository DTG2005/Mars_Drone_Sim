import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range, LaserScan, Imu, FluidPressure
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64, Float64MultiArray
from rclpy.qos import QoSProfile
import math

class MarsRoverPIDController(Node):
    def __init__(self):
        super().__init__('mars_rover_pid_controller')

        # Parameters for PID gains and target height
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.3)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('target_height', 10.0)

        # Initialize PID parameters
        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        self.target_height = self.get_parameter('target_height').get_parameter_value().double_value
        self.current_height = 0.0

        # Simulation parameters
        self.current_height = 5.0  # Initial height
        self.velocity = 0.0  # Simulated velocity
        self.mass = 1.0  # Mass of the rover
        self.damping = 0.1  # Damping coefficient for air resistance
        self.martian_gravity = 3.721  # Martian gravity (m/s^2)

        # PID state variables
        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()

        # Timer for PID control loop (runs every 0.1s)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Publishers and subscriptions
        self.thrust_publisher = self.create_publisher(Wrench, '/mav/thrust', QoSProfile(depth=10))
        self.sensor_subscription = self.create_subscription(
            Float64,
            '/rover/altitude_sensor',
            self.height_callback,
            10,
        )
        self.imu_subscription = self.create_subscription(
            Imu,
            '/rover/imu',
            self.imu_callback,
            10,
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/rover/lidar',
            self.lidar_callback,
            10,
        )
        self.barometer_subscription = self.create_subscription(
            FluidPressure,
            '/rover/barometer',
            self.barometer_callback,
            10,
        )
        self.compass_subscription = self.create_subscription(
            Float64,
            '/rover/compass',
            self.compass_callback,
            10,
        )
        self.radar_subscription = self.create_subscription(
            Float64MultiArray,
            '/rover/radar',
            self.radar_callback,
            10,
        )

        # Additional state variables
        self.heading = 0.0  # Compass heading (0-360 degrees)
        self.radar_distances = []  # List to hold radar distances

        self.get_logger().info("MarsRoverPIDController Node has been started.")

    def height_callback(self, msg):
        """Update the current height from sensor data."""
        self.current_height = msg.data
        self.get_logger().info(f"Received sensor height: {self.current_height:.2f}")

    def imu_callback(self, msg):
        """Process IMU data."""
        q = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(q.x, q.y, q.z, q.w)
        self.get_logger().info(f"IMU - Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, "
                               f"Linear Accel Z: {msg.linear_acceleration.z:.2f}")

    def lidar_callback(self, msg):
        """Process LiDAR data."""
        if msg.ranges:
            min_distance = min(msg.ranges)
            self.get_logger().info(f"LiDAR Minimum Distance: {min_distance:.2f}")

    def barometer_callback(self, msg):
        """Process Barometer data."""
        sea_level_pressure = 101325.0  # Pa
        pressure = msg.fluid_pressure
        self.barometric_height = (1 - (pressure / sea_level_pressure) ** (1 / 5.255)) * 44330.77
        self.get_logger().info(f"Barometric Height: {self.barometric_height:.2f} m")

    def compass_callback(self, msg):
        """Process Compass data."""
        self.heading = msg.data  # Heading in degrees
        self.get_logger().info(f"Compass Heading: {self.heading:.2f}°")

    def radar_callback(self, msg):
        """Process Radar data."""
        self.radar_distances = msg.data
        if self.radar_distances:
            self.get_logger().info(f"Radar Distances: {self.radar_distances}")

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))  # Clamp value to avoid NaN
        pitch = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def control_loop(self):
        """Periodic PID control loop."""
        current_time = self.get_clock().now()
        time_delta = (current_time - self.last_time).nanoseconds / 1e9  # Convert to seconds
        if time_delta <= 0.0:
            return  # Avoid division by zero or invalid time step

        # Compute PID errors
        error = self.target_height - self.current_height
        self.error_sum += error * time_delta

        # Anti-Windup: Limit the integral term to prevent unbounded growth
        max_integral = 100.0  # Adjust based on system requirements
        self.error_sum = max(min(self.error_sum, max_integral), -max_integral)

        d_error = (error - self.last_error) / time_delta

        # PID control law
        thrust = self.kp * error + self.ki * self.error_sum + self.kd * d_error
        thrust = max(0.0, thrust)  # Ensure non-negative thrust

        # Update the simulation model (simulate feedback)
        self.simulate_physics(thrust, time_delta)

        # Publish thrust command
        self.publish_thrust(thrust)

        # Update state
        self.last_error = error
        self.last_time = current_time

        # Log for debugging
        self.get_logger().info(f"Height: {self.current_height:.2f}, "
                            f"Error: {error:.2f}, "
                            f"P: {self.kp * error:.2f}, "
                            f"I: {self.ki * self.error_sum:.2f}, "
                            f"D: {self.kd * d_error:.2f}, "
                            f"Thrust: {thrust:.2f}")
        
    def simulate_physics(self, thrust, time_delta):
        """Simulate feedback from the environment including Martian gravity."""
        # Calculate acceleration (F = ma, considering damping force and gravity)
        gravitational_force = self.mass * self.martian_gravity
        net_force = thrust - gravitational_force - self.damping * self.velocity
        acceleration = net_force / self.mass

        # Update velocity and height
        self.velocity += acceleration * time_delta
        self.current_height += self.velocity * time_delta

        # Ensure height remains non-negative
        self.current_height = max(0.0, self.current_height)

    def publish_thrust(self, thrust):
        """Publish the computed thrust command."""
        wrench = Wrench()
        wrench.force.z = thrust
        self.thrust_publisher.publish(wrench)


def main(args=None):
    rclpy.init(args=args)
    node = MarsRoverPIDController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()