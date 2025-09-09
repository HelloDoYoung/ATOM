import cv2
import numpy as np

class CameraOpen:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print('failed to open camera')
            exit()

    def gaussian_blur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (0, 0), sigmaX=2, sigmaY=2)
        return gaussian_src

    def main(self):
        _, src = self.cap.read()
        gaussian_src = self.gaussian_blur(src)
        cv2.imshow('src', src)
        cv2.imshow('gaussian_blur', gaussian_src)

if __name__ == '__main__':
    node = CameraOpen()
    while(1):
        node.main()
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()