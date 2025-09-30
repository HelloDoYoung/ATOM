import cv2
import numpy as np

class LaneDetection():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print('failed to open camera')
            exit()
        
    def sobel_xy(self, src):
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
        scale_factor = np.max(gradmag)/255  
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        th_mag = (100, 255)
        gradient_magnitude = np.zeros_like(gradmag)
        gradient_magnitude[(gradmag >= th_mag[0]) & (gradmag <= th_mag[1])] = 255
        return gradient_magnitude
    
    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src

    def morphology(self, src):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        src = cv2.dilate(src, kernel)
        return src
    
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
        sums, count = (0,0)
        for contour in contours:
            if cv2.contourArea(contour) < 100: continue
            m = cv2.moments(contour)
            sums += int(m['m10']/m['m00'])
            count += 1
        try: sums = sums//count
        except: sums = 320
        return src, sums
    
    def main(self):
        _, src = self.cap.read()
        src = self.perspectiveTransformation(src)
        src2, center = self.findLaneCenter(src)
        angle = (320-center)/320
        cv2.circle(src, (center,60), 5, (255,0,0), -1)
        cv2.imshow('src', src)
        cv2.imshow('src2', src2)
        
if __name__ == '__main__':
    node = LaneDetection()
    while(1):
        node.main()
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()
