import cv2
import numpy as np

# 트랙바에서 사용하는 함수
def nothing(x):
    pass

# 창과 트랙바 생성
cv2.namedWindow('HSV Trackbars')

# 트랙바 생성 (H, S, V의 최소, 최대 값을 각각 조정 가능)
cv2.createTrackbar('H Min', 'HSV Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Trackbars', 179, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Trackbars', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Trackbars', 0, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Trackbars', 255, 255, nothing)

# 웹캠에서 비디오 스트림 시작
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 트랙바에서 HSV 값 가져오기
    h_min = cv2.getTrackbarPos('H Min', 'HSV Trackbars')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Trackbars')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Trackbars')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Trackbars')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Trackbars')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Trackbars')

    # HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 트랙바 값으로 마스크 생성
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 마스크를 이용해 원본 이미지에서 해당 색상만 추출
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # 결과 화면 출력
    cv2.imshow('Original', frame)
    cv2.imshow('Masked', masked_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 스트림 해제
cap.release()
cv2.destroyAllWindows()
