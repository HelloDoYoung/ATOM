import cv2
import numpy as np

class Moments:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.start = False
        if not self.cap.isOpened():
            print('failed to open camera')
            exit()

    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src

    def redHsvInrange(self, src):
        red_lower_bound = np.array([0, 150, 150], dtype=np.uint8)
        red_upper_bound = np.array([20, 255, 255], dtype=np.uint8)
        red_lower_bound2 = np.array([160, 150, 150], dtype=np.uint8)
        red_upper_bound2 = np.array([179, 255, 255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst1 = cv2.inRange(hsv_src, red_lower_bound, red_upper_bound)
        hsv_dst2 = cv2.inRange(hsv_src, red_lower_bound2, red_upper_bound2)
        hsv_dst = hsv_dst1 | hsv_dst2
        return hsv_dst

    def greenHsvInrange(self, src):
        green_lower_bound = np.array([30, 150, 150], dtype=np.uint8)
        green_upper_bound = np.array([90, 255, 255], dtype=np.uint8)
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
        if np.count_nonzero(src) > 4000:
            return True
        return False

    def checkGreenLight(self, src):
        if np.count_nonzero(src) > 4000:
            return True
        return False

    def main(self):
        _, src = self.cap.read()
        src = self.gaussianBlur(src)
        src_red = self.redHsvInrange(src)
        src_green = self.greenHsvInrange(src)
        src_red = self.componentsWithStatsFilter(src_red)
        src_green = self.componentsWithStatsFilter(src_green)
        if self.checkRedLight(src_red):
            print('Stop')
        elif self.checkGreenLight(src_green):
            print('Start')
        cv2.imshow('src_red', src_red)
        cv2.imshow('src_green', src_green)
        cv2.imshow('src', src)

if __name__ == '__main__':
    node = Moments()
    while(1):
        node.main()
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()
