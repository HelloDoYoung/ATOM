import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage

class LaneDetection(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
        self.laneModeSubscriber = self.create_subscription(
            Bool,
            '/lane_mode',
            self.laneModeSubscriber_callback,
            10
        )
        self.stateSubscriber = self.create_subscription(
            Bool,
            '/lane_state',
            self.stateSubscriber_callback,
            10
        )
        self.velocityPublisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        self.intersectionStateSubscriber = self.create_subscription(
            Bool,
            '/intersection_state',
            self.intersectionStateSubscriber_callback,
            10
        )
        self.lane_mode = True  # default = True(yellow)
        self.state = False
        self.is_intersection = False
    
    def intersectionStateSubscriber_callback(self, msg):
        if not self.is_intersection and msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Start Intersection Mode')
        if self.is_intersection and not msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Finish Intersection Mode')
    
    def stateSubscriber_callback(self, msg):
        self.state = msg.data
        if self.state == True: self.get_logger().info('Start Lane Detection')
        else:
            twist = Twist()
            self.velocityPublisher.publish(twist)
            self.get_logger().info('Terminate Lane Detection')
    
    def sobel_xy(self, src):
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
        scale_factor = np.max(gradmag)/255  
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        th_mag = (100, 255)  #(30, 255)
        gradient_magnitude = np.zeros_like(gradmag)
        gradient_magnitude[(gradmag >= th_mag[0]) & (gradmag <= th_mag[1])] = 255
        return gradient_magnitude
    
    def morphology(self, src):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        src = cv2.dilate(src, kernel)
        return src
    
    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (5,5), sigmaX=0, sigmaY=0)
        return gaussian_src
    
    def componentsWithStatsFilter(self, src):
        min_area = 800
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        mask = np.isin(labels, valid_labels)
        filtered = (mask * 255).astype(np.uint8)
        return filtered
    
    def perspectiveTransformation(self, src):
        src_ptr = np.float32([[90,360],[550,360],[640,480],[0,480]])
        dst_ptr = np.float32([[20,0],[620,0],[640,120],[0,120]])
        mtrx = cv2.getPerspectiveTransform(src_ptr, dst_ptr)
        src = cv2.warpPerspective(src, mtrx, (640, 120))
        return src
   
    def findLaneCenter(self, src):
        src = self.gaussianBlur(src)
        src = self.sobel_xy(src)
        src = self.morphology(src)
        src = self.componentsWithStatsFilter(src)
        contours, _ = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        sums = 0
        count = 0
        if np.count_nonzero(src) < len(src)*len(src[0])//4:
            for contour in contours:
                if cv2.contourArea(contour) < 100: continue
                m = cv2.moments(contour)
                sums += int(m['m10']/m['m00'])
                count += 1
        try:
            if self.lane_mode: sums = sums//count
            else: sums = sums//count + 320
        except:
            if self.lane_mode: sums = 0
            else: sums = 640
        return src, sums
    
    def laneModeSubscriber_callback(self, msg):
        self.lane_mode = msg.data
        if self.lane_mode: self.get_logger().info('Change lane mode to YELLOW')
        else: self.get_logger().info('Change lane mode to WHITE')
        cv2.destroyAllWindows()
        return
    
    def videoSubscriber_callback(self, msg):
        if not self.state: return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            src_yellow = np.empty((120,320))
            src_white = np.empty((120,320))
            src = self.perspectiveTransformation(src)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return
        
        if self.lane_mode:
            src_yellow, center_yellow = self.findLaneCenter(src[:,:320])
            twist = Twist()
            angle = (70-center_yellow)/70
            if angle < -1.5: angle = -1.5
            elif angle > 1.5: angle = 1.5
            if self.is_intersection:
                twist.linear.x = 0.1 / (1 + abs(angle)/1.5)
                twist.angular.z = angle/2
            else:
                twist.linear.x = 0.22 / (1 + abs(angle)/1.5)
                twist.angular.z = angle
            twist.angular.z = angle
            self.velocityPublisher.publish(twist)
            cv2.circle(src, (center_yellow,60), 5, (0,0,255), -1)
            cv2.imshow('src_yellow', src_yellow)
            
        else:
            src_white, center_white = self.findLaneCenter(src[:,320:])
            twist = Twist()
            angle = (570-center_white)/70
            if angle < -1.5: angle = -1.5
            elif angle > 1.5: angle = 1.5
            if self.is_intersection:
                twist.linear.x = 0.1 / (1 + abs(angle)/1.5)
                twist.angular.z = angle/2
            else:
                twist.linear.x = 0.22 / (1 + abs(angle)/1.5)
                twist.angular.z = angle
            self.velocityPublisher.publish(twist)
            cv2.circle(src, (center_white,60), 5, (255,0,0), -1)
            cv2.imshow('src_white', src_white)
        
        cv2.imshow('src', src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            twist = Twist()
            self.velocityPublisher.publish(twist)
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = LaneDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
