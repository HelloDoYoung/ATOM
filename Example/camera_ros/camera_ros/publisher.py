import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import pickle
import os
import numpy as np

class UndistortedVideoPublisher(Node):
    def __init__(self):
        super().__init__('undistorted_video_publisher')

        # Publisher
        self.pub = self.create_publisher(CompressedImage, 'camera', 10)

        # Load camera calibration
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        calib_file = os.path.join(BASE_DIR, 'camera_calibration.pkl')
        if not os.path.exists(calib_file):
            self.get_logger().error(f"Calibration file not found: {calib_file}")
            return

        with open(calib_file, 'rb') as f:
            calib_data = pickle.load(f)
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeffs']

        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera.")
            return

        # Prepare undistortion map
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab a frame for calibration mapping.")
            return
        h, w = frame.shape[:2]

        # alpha=0 -> black area removal
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h)
        )

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2
        )

        # Timer ~30 FPS
        self.timer = self.create_timer(0.03, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame.")
            return

        # Undistort
        undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        # Crop to ROI to remove black borders
        x, y, w_crop, h_crop = self.roi
        undistorted = undistorted[y:y+h_crop, x:x+w_crop]
        
        '''
        # Show undistorted video
        cv2.imshow('Undistorted Video', undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return
        '''
        
        # Publish as CompressedImage
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        ret, buffer = cv2.imencode('.jpg', undistorted)
        if not ret:
            self.get_logger().warn("Failed to encode frame.")
            return
        msg.data = np.array(buffer).tobytes()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UndistortedVideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

