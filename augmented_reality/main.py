import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from objloader_simple import OBJ
from utils import render, projection_3D
import copy

def run_ar():
    # load query image
    query = cv2.imread("src/cards.jpg")
    height, width = query.shape[:2]
    # load camera frame
    self_cap = cv2.VideoCapture("src/video.mov")
    rgb = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    # load the 3d object
    obj_3d = OBJ('3d_objects/fox.obj', swapyz=True)

    # declare sift object detector
    sift = cv2.SIFT_create()

    # initlize empty rotated matrix for rotation later
    rotated_matrix = []

    # start the ar projection
    while True:
        detect = False
        prev_time = time.time()
        try:
            success, frame = self_cap.read()
        except:
            print("Failed to read the file")

        if frame is None: print("Process ends")

        # point and object detection
        query_kp, query_des = sift.detectAndCompute(rgb,None)
        train_kp, train_des = sift.detectAndCompute(frame.astype(np.uint8),None)

        match_points = []
        bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
        match = bf.knnMatch(query_des,train_des,k=2)
        # ratio test
        ratio = 0.5
        for ptn1,ptn2 in match:
            if ptn1.distance < ratio * ptn2.distance:
                match_points.append(ptn1)

        print("Projecting....")
        # at least 4 points for projection
        if len(match_points) >= 4:
            detect = True
            set_point1 = np.float32([query_kp[ptn.queryIdx].pt for ptn in match_points]).reshape((-1, 1, 2))
            set_point2 = np.float32([train_kp[ptn.trainIdx].pt for ptn in match_points]).reshape((-1, 1, 2))
            homography_matrix, _ = cv2.findHomography(set_point1,set_point2,cv2.RANSAC,5)

            if homography_matrix is not None:
                homography_matrix_3d = projection_3D(homography_matrix)
                rotated_matrix, frame = render(frame, obj_3d, homography_matrix_3d, query, rotated_matrix, prev_time)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ =='__main__':
    run_ar()
