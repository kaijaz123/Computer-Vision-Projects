import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use("dark_background")

def run_ar():
    query = cv2.imread("cards.jpg")
    height, width = query.shape[:2]

    self_cap = cv2.VideoCapture("video.mov")
    kanna_cap = cv2.VideoCapture("kanna.mp4")
    rgb = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    # declare sift object detector
    sift = cv2.SIFT_create()

    # start the ar projection
    while True:
        detect = False
        try:
            success, frame = self_cap.read()
            success, kanna = kanna_cap.read()
        except:
            print("Failed to read the video file")

        if frame is None: print("Video ends")
        resize_kanna = cv2.resize(kanna, (width,height))

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
        # create a mask of frame
        mask = frame.copy()

        if len(match_points) >= 4:
            detect = True
            # reshape into 3d
            set_point1 = np.float32([query_kp[ptn.queryIdx].pt for ptn in match_points]).reshape((-1, 1, 2))
            set_point2 = np.float32([train_kp[ptn.trainIdx].pt for ptn in match_points]).reshape((-1, 1, 2))
            matrix, _ = cv2.findHomography(set_point1,set_point2,cv2.RANSAC,5)

            # look for hormography and warp the point
            pts = np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,matrix)
            imgwrap = cv2.warpPerspective(resize_kanna,matrix, (frame.shape[1],frame.shape[0]))

            # overlay the wrapped image to frame
            roi_x, roi_y = np.where(np.all(imgwrap != [0,0,0], axis = -1))
            mask[roi_x,roi_y] = imgwrap[roi_x,roi_y]

            # put Text
            cv2.putText(mask, "Project Region Found: {}".format(detect), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),
                        2, cv2.LINE_AA)

        if not detect:
            cv2.putText(mask, "Project Region Found: {}".format(detect), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),
                        2, cv2.LINE_AA)

        cv2.imshow('Projection', mask)
        cv2.imshow("Original Frame", frame)
        cv2.waitKey(1)


if __name__ =='__main__':
    run_ar()
