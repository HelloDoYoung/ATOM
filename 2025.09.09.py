import cv2
import numpy as np

class CameraOpen:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            print('failed to open camera')
            exit()

    def gaussian_blur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (0, 0), sigmaX=2, sigmaY=2)
        return gaussian_src

    def redHsvInRange(self, src):
        red_lower_bound = np.array([0, 150, 150], dtype=np.uint8)
        red_upper_bound = np.array([20, 255, 255], dtype=np.uint8)
        red_lower_bound2 = np.array([160, 150, 150], dtype=np.uint8)
        red_upper_bound2 = np.array([179, 255, 255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, red_lower_bound, red_upper_bound)
        hsv_dst2 = cv2.inRange(hsv_src, red_lower_bound2, red_upper_bound2)
        hsv_dst3 = hsv_dst | hsv_dst2
        return hsv_dst3

    def main(self):
        _, src = self.cap.read()
        gaussian_src = self.gaussian_blur(src)
        hsv_red_src = self.redHsvInRange(gaussian_src)
        cv2.imshow('src', src)
        cv2.imshow('gaussian_blur', gaussian_src)
        cv2.imshow('hsv_red_src', hsv_red_src)

if __name__ == '__main__':
    node = CameraOpen()
    while(1):
        node.main()
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()