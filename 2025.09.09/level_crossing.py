import cv2
import numpy as np

class LevelCrossing:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print('failed to open camera')
            exit()

    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src

    def redHsvInrange(self, src):
        red_lower_bound = np.array([0,120,120], dtype=np.uint8)
        red_upper_bound = np.array([15,255,255], dtype=np.uint8)
        red_lower_bound2 = np.array([165,120,120], dtype=np.uint8)
        red_upper_bound2 = np.array([179,255,255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, red_lower_bound, red_upper_bound)
        hsv_dst2 = cv2.inRange(hsv_src, red_lower_bound2, red_upper_bound2)
        hsv_dst3 = hsv_dst | hsv_dst2
        return hsv_dst3

    def componentsWithStatsFilter(self, src):
        min_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        mask = np.isin(labels, valid_labels)
        filtered = (mask * 255).astype(np.uint8)
        return filtered

    def findRedMoments(self, src):
        centers = []
        contours, hierachy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 2000: continue
            m = cv2.moments(contour)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            centers.append((cx, cy))
        return centers

    def check_barrier(self, points):
        slope = []
        inf_count = 0
        if len(points) >= 2:
            for i in range(len(points) - 1):
                dx = points[i][0] - points[i+1][0]
                dy = points[i][1] - points[i+1][1]
                if dx == 0: inf_count += 1
                else: slope.append(abs(dy/dx))
            if inf_count > 0: return True
            if slope:
                slope_avg = sum(slope)/len(slope)
                if slope_avg > 1: return True
        return False
    
    def main(self):
        _, src = self.cap.read()
        src2 = self.gaussianBlur(src)
        src2 = self.redHsvInrange(src2)
        src2 = self.componentsWithStatsFilter(src2)
        moments = self.findRedMoments(src2)
        if self.check_barrier(moments) is True: print('Open')
        for m in moments:
            cv2.circle(src, m, 5, (255,100,100), -1)
        cv2.imshow('src', src)
        cv2.imshow('src2', src2)

if __name__ == '__main__':
    node = LevelCrossing()
    while(1):
        node.main()
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()
