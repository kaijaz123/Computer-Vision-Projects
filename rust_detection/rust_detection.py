import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

rust_1 = 'rust_1.png'
rust_2 = 'rust_2.png'
nrust = 'no_rust.jpg'

def rust(*args):
    for image in args:
        found = False
        img = cv2.imread(image)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilation = cv2.dilate(binary, se, iterations = 1)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, se) # remove small blob

        contours, hierarchy = cv2.findContours(opening, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
        maxAT = 10000 # maximum area threshold
        minAT = 2000 # minimum area threshold
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area <= maxAT and area >= minAT:
                found = True
                largest_area=area
                largest_contour_index=index
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(rgb, (x,y), (x+w, y+h), color = (255,0,0), thickness = 1)

        if not found:
            cv2.putText(rgb, "No rust detected", (20,20), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.5, color = (255,0,0), thickness = 2)
        plt.imshow(rgb)
        plt.show()


if __name__ == '__main__':
    rust(rust_1,rust_2,nrust)
