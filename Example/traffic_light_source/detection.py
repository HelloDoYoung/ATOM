import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class TrafficLight(Node):
    def __init__(self):
        super().__init__('traffic_light_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
        
    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src

    def redHsvInrange(self, src):
        red_lower_bound = np.array([0,120,120], dtype=np.uint8)
        red_upper_bound = np.array([15,255,255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, red_lower_bound, red_upper_bound)
        return hsv_dst

    def greenHsvInrange(self, src):
        green_lower_bound = np.array([40,100,150], dtype=np.uint8)
        green_upper_bound = np.array([75,255,255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, green_lower_bound, green_upper_bound)
        return hsv_dst

    def componentsWithStatsFilter(self, src):
        min_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        mask = np.isin(labels, valid_labels)
        filtered = (mask * 255).astype(np.uint8)
        return filtered

    def checkRedLight(self, src):
        threshold = 4000
        if np.count_nonzero(src) > threshold:
            return True
        return False

    def checkGreenLight(self, src):
        threshold = 4000
        if np.count_nonzero(src) > threshold:
            return True
        return False

    def videoSubscriber_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return
            
        src2 = self.gaussianBlur(src)
        src_red = self.redHsvInrange(src2)
        src_green = self.greenHsvInrange(src2)
        src_red = self.componentsWithStatsFilter(src_red)
        src_green = self.componentsWithStatsFilter(src_green)
        if self.checkRedLight(src_red): print('Stop')
        elif self.checkGreenLight(src_green): print('Start')
        cv2.imshow('src', src)
        cv2.imshow('src_red', src_red)
        cv2.imshow('src_green', src_green)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = TrafficLight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
