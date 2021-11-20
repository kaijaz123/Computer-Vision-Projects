import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def GLCM(roi, image, display_img):
    pixel_value = (0,255,0)
    properties = []
    count = 1
    for contour in roi:
        copy = image.copy()
        mask = np.zeros_like(image)
        x,y,w,h = cv2.boundingRect(contour[0])
        cv2.drawContours(mask, contour, -1, pixel_value, lineType = cv2.LINE_AA, thickness = cv2.FILLED)
        x,y,_ = np.where(mask != pixel_value)
        copy[x,y] = (0,0,0)

        # cropped the image
        x,y,w,h = cv2.boundingRect(contour[0])
        cropped = copy[y:y+h,x:x+w]
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # perform glcm
        # check for contrast - if high value means clear (cause can see the object under water clearly)
        # low value - means muddy water (all pixels are almost same and we cant see the object under water clealy)
        glcm = greycomatrix(cropped_gray, [1], [0], 256, symmetric=True, normed=True)
        value = greycoprops(glcm, "contrast")[0][0]

        centroid = (((x+w)-x)//2 + x - 50, ((y+h)-y)//2 + y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,0,0)
        scale = 0.5
        thickness = 2
        if value < 50:
            cv2.putText(display_img, "muddy water", centroid, fontFace = font ,color = (0,255,0),
                        fontScale = scale, thickness = thickness)
        else:
            cv2.putText(display_img, "water clear", centroid, fontFace = font ,color = color,
                        fontScale = scale, thickness = thickness)

    cv2.putText(display_img, "There are {} pond areas".format(len(roi)),
               (display_img.shape[1] - 200, display_img.shape[0] - 10), fontFace = font,
               color = color, fontScale = 0.5, thickness = 2)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    return display_img

def segmentation(video):
    cam = cv2.VideoCapture(video)
    finalized_mask = None
    contour_cnt = 0
    ROI = []

    while True:
        success, frame = cam.read()
        if frame is None : break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if finalized_mask is None:
            finalized_mask = np.zeros((int(frame.shape[0]), int(frame.shape[1]), 3))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            mask = np.zeros_like(gray)
            blur = cv2.medianBlur(gray, 3)
            canny = cv2.Canny(blur, 100, 200)
            lines = cv2.HoughLines(canny, 2, np.pi/180, 200)
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(mask,(x1,y1), (x2,y2), (255,255,255),2)

            canny = cv2.Canny(mask, 100, 200)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            dilation = cv2.dilate(canny, se, 1)
            reverse = cv2.bitwise_not(dilation)

            contours, hir = cv2.findContours(reverse, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for index,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                #color = np.random.randint(255, size = (3,))
                #color = (int(color[0]), int(color[1]), int(color[2]))
                arc = cv2.arcLength(contour, True)
                if area < 15000: continue
                contour_cnt += 1
                cv2.drawContours(finalized_mask, contours, index, (255,0,0), lineType = cv2.LINE_AA, thickness = 2)

                # append the ROI position
                ROI.append([contour])

        region = np.where(finalized_mask > 1)
        segment_x, segment_y = region[0], region[1]
        rgb[segment_x,segment_y] = (255,0,0)
        rgb = GLCM(ROI,frame, rgb)

        cv2.imshow("Segmented Pond - (Muddy or Clear)", rgb)
        cv2.waitKey(1)

if __name__ == '__main__':
    video_name = 'water_pond.avi'
    plt.style.use("dark_background")
    segmentation(video_name)
