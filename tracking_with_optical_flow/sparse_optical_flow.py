import cv2
import matplotlib.pyplot as plt
import numpy as np

video = 'cactus.avi'
plt.style.use("dark_background")
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cam = cv2.VideoCapture(video)
_, first_frame = cam.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)

# get the desire track point
_, binary = cv2.threshold(prev_gray, 130, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
la = 0
for index,contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > la:
        la = area
        i = index
        x, y, w, h = cv2.boundingRect(contour)
        cw = ((x+w) - x) // 2
        ch = ((y+h) - y) // 2
        centroid = np.array([[np.float32(cw+x),np.float32(ch+y)]])

while True:
    success, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_centroid, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, centroid, None, **lk_params)
    for index, (new, old) in enumerate(zip(new_centroid, centroid)):
        old_x, old_y = new.ravel()
        new_x, new_y = old.ravel()
        mask = cv2.line(mask, (int(old_x), int(old_y)), (int(new_x), int(new_y)), (255,0,0), 2)
        frame = cv2.circle(frame, (int(old_x), int(old_y)), 3, (255,0,0), -1)

    output = cv2.add(frame, mask)
    prev_gray = gray.copy()
    centroid = new_centroid
    cv2.imshow("Output", output)
    cv2.waitKey(1)












    #a
