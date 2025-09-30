import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist

class FrontBackNode(Node):
    def __init__(self):
        super().__init__('front_back_node')
        self.velocityPublisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(10, self.timer_callback)
        self.direction = 1

    def timer_callback(self):
        twist = Twist()
        twist.linear.x = 0.1 * self.direction
        self.velocityPublisher.publish(twist)
        self.get_logger().info(f'Published :\n{twist}')
        self.direction *= -1

def main(args=None):
    rclpy.init(args=args)
    node = FrontBackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt as e:
        node.get_logger().info(f'Keyboard Interrupt {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()