#!/usr/bin/python3
import cv2
import numpy as np

#################### KULLANICI AYARLARI ####################
downSample = 1.0            # Çözünürlüğü düşürmek için < 1.0 yapabilirsin (örn. 0.5)
zoomFactor = 0.9            # Kenarları kırpmak için
processVar = 0.03
measVar = 2
roiDiv = 3.5
showFullScreen = 0
showrectROI = 0
showTrackingPoints = 0
showUnstabilized = 0
maskFrame = 0
delay_time = 1

#################### Kamera Kaynağı #########################
camera_index = 1  # Laptop kamerası genelde 0’dır. Harici USB kamera 1 olabilir.
video = cv2.VideoCapture(camera_index)
if not video.isOpened():
    print("Kamera açılamadı!")
    exit()

lk_params = dict(winSize=(15,15), maxLevel=3, 
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

count = 0
a = x = y = 0
Q = np.array([[processVar]*3])
R = np.array([[measVar]*3])
prevFrame = None

while True:
    grab, frame = video.read()
    if not grab:
        break

    res_w_orig = frame.shape[1]
    res_h_orig = frame.shape[0]
    res_w = int(res_w_orig * downSample)
    res_h = int(res_h_orig * downSample)

    top_left = [int(res_h/roiDiv), int(res_w/roiDiv)]
    bottom_right = [int(res_h - res_h/roiDiv), int(res_w - res_w/roiDiv)]

    Orig = frame.copy()
    if downSample != 1:
        frame = cv2.resize(frame, (res_w, res_h))

    currFrame = frame
    currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
    currGray = currGray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    if prevFrame is None:
        prevOrig = frame
        prevGray = currGray
        prevFrame = frame
        continue

    if showrectROI:
        cv2.rectangle(prevOrig, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (211,211,211), 1)

    prevPts = cv2.goodFeaturesToTrack(prevGray, maxCorners=400, qualityLevel=0.01, minDistance=30, blockSize=3)

    if prevPts is not None:
        currPts, status, err = cv2.calcOpticalFlowPyrLK(prevGray, currGray, prevPts, None, **lk_params)
        idx = np.where(status == 1)[0]

        prevPts = prevPts[idx] + np.array([int(res_w_orig/roiDiv), int(res_h_orig/roiDiv)])
        currPts = currPts[idx] + np.array([int(res_w_orig/roiDiv), int(res_h_orig/roiDiv)])

        if showTrackingPoints:
            for pT in prevPts:
                cv2.circle(prevOrig, (int(pT[0][0]), int(pT[0][1])), 5, (211,211,211), -1)

        if prevPts.size and currPts.size:
            m, _ = cv2.estimateAffinePartial2D(prevPts, currPts)
        else:
            m = lastRigidTransform if 'lastRigidTransform' in locals() else np.eye(2,3)

        if m is None:
            m = lastRigidTransform

        dx, dy = m[0,2], m[1,2]
        da = np.arctan2(m[1,0], m[0,0])
    else:
        dx = dy = da = 0

    x += dx
    y += dy
    a += da
    Z = np.array([[x, y, a]], dtype="float")

    if count == 0:
        X_estimate = np.zeros((1,3), dtype="float")
        P_estimate = np.ones((1,3), dtype="float")
    else:
        X_predict = X_estimate
        P_predict = P_estimate + Q
        K = P_predict / (P_predict + R)
        X_estimate = X_predict + K * (Z - X_predict)
        P_estimate = (np.ones((1,3), dtype="float") - K) * P_predict

    diff_x = X_estimate[0,0] - x
    diff_y = X_estimate[0,1] - y
    diff_a = X_estimate[0,2] - a
    dx += diff_x
    dy += diff_y
    da += diff_a

    m = np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da),  np.cos(da), dy]
    ])

    fS = cv2.warpAffine(prevOrig, m, (res_w_orig, res_h_orig))
    T = cv2.getRotationMatrix2D((res_w_orig/2, res_h_orig/2), 0, zoomFactor)
    f_stabilized = cv2.warpAffine(fS, T, (res_w_orig, res_h_orig))

    if maskFrame:
        mask = np.zeros(f_stabilized.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (100, 200), (1180, 620), 255, -1)
        f_stabilized = cv2.bitwise_and(f_stabilized, f_stabilized, mask=mask)

    window_name = 'Kamera Stabilizasyon'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if showFullScreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(window_name, f_stabilized)
    if showUnstabilized:
        cv2.imshow("Unstabilized ROI", prevGray)

    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

    prevOrig = Orig
    prevGray = currGray
    prevFrame = currFrame
    lastRigidTransform = m
    count += 1

video.release()
cv2.destroyAllWindows()
