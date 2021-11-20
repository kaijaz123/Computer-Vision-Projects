import cv2
import numpy as np
import matplotlib.pyplot as plt

day = "Q1_day_video.avi"
night = "Q1_night_video.avi"
plt.style.use("dark_background")

def day_night(day_video, night_video):
    day_vid = cv2.VideoCapture(day_video)
    night_vid = cv2.VideoCapture(night_video)
    while True:
        success, day_frame = day_vid.read()
        success, night_frame = night_vid.read()
        day_gray = cv2.cvtColor(day_frame, cv2.COLOR_BGR2GRAY)
        night_gray = cv2.cvtColor(night_frame, cv2.COLOR_BGR2GRAY)
        ret, day_binary = cv2.threshold(day_gray, 150, 255, cv2.THRESH_BINARY_INV)
        ret, night_binary = cv2.threshold(night_gray, 180, 255, cv2.THRESH_BINARY)

        # day
        day_sobelx = cv2.Sobel(day_gray, cv2.CV_8U, 1, 2)
        day_sobely = cv2.Sobel(day_gray, cv2.CV_8U, 2, 1)
        day_combined = cv2.addWeighted(day_sobelx, 1, day_sobely, 2, 2)
        day_result = cv2.bitwise_not(day_binary, day_combined, mask = None)

        # night
        ret, night_binary = cv2.threshold(night_gray, 140, 255, cv2.THRESH_BINARY)
        contours, hir = cv2.findContours(night_binary, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
        la = 0
        for index,contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > la:
                la = area
                i = index

        cv2.drawContours(night_binary, contours, i, color = 0, lineType = cv2.LINE_AA, thickness = cv2.FILLED)

# Please set the window accordingly to your resolution
#        cv2.namedWindow("Day_ori")
#        cv2.namedWindow("Day_result")
#        cv2.namedWindow("Night_ori")
#        cv2.namedWindow("Night_result")
#        cv2.moveWindow("Day_ori", 1940, 50)
#        cv2.moveWindow("Day_result", 2450, 50)
#        cv2.moveWindow("Night_ori", 2960, 50)
#        cv2.moveWindow("Night_result", 3470, 50)
        cv2.imshow("Day_ori", day_frame)
        cv2.imshow("Day_result", day_result)
        cv2.imshow("Night_ori", night_frame)
        cv2.imshow("Night_result", night_binary)
        cv2.waitKey(1)

if __name__ == "__main__":
    day_night(day,night)
