import cv2
import numpy as np

class IntersectionDetection:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print('failed to open camera')
            exit()

    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9, 9), sigmaX=0, sigmaY=0)
        return gaussian_src
    
    def detectRL(self, src):
        gray = self.gaussianBlur(src)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=200,
            param1=100,
            param2=60,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(src, (x, y), r, (0, 255, 0), 2)
                cv2.circle(src, (x, y), 3, (0, 0, 255), -1)

                try:
                    left = (int(x - r / 3), int(y + r / 3))
                    right = (int(x + r / 3), int(y + r / 3))
                    
                    cv2.circle(src, left, 3, (255, 50, 50), -1)
                    cv2.circle(src, right, 3, (255, 150, 150), -1)
                    
                    if gray[left[1], left[0]] > gray[right[1], right[0]]:
                        return src, -1 
                    elif gray[left[1], left[0]] < gray[right[1], right[0]]:
                        return src, 1

                except IndexError:
                    print('\n**out of frame**\n')
            
            return src, 0
        
        return src, 0

    def main(self):
        ret, src = self.cap.read()
        if not ret:
            return

        src, result = self.detectRL(src)
        
        if result == 1:
            print('left sign detected')
        elif result == -1:
            print('right sign detected')
            
        cv2.imshow('src', src)

if __name__ == '__main__':
    node = IntersectionDetection()
    while True:
        node.main()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()