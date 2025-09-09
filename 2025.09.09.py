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

    def main(self):
        _, src = self.cap.read()
        cv2.imshow('src', src)

if __name__ == '__main__':
    node = CameraOpen()
    while(1):
        node.main()
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()