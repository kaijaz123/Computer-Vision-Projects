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
    # matrix of camera parameters
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # declare sift object detector
    sift = cv2.SIFT_create()

    # start the ar projection
    while True:
        detect = False
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
        if len(match_points) >= 4:
            detect = True
            # reshape into 3d
            set_point1 = np.float32([query_kp[ptn.queryIdx].pt for ptn in match_points]).reshape((-1, 1, 2))
            set_point2 = np.float32([train_kp[ptn.trainIdx].pt for ptn in match_points]).reshape((-1, 1, 2))
            homography_matrix, _ = cv2.findHomography(set_point1,set_point2,cv2.RANSAC,5)

            if homography_matrix is not None:
                homography_matrix_3d = projection_3D(camera_parameters, homography_matrix)
                frame = render(frame, obj_3d, homography_matrix_3d, query)

        scale_movement += 20
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ =='__main__':
    run_ar()
