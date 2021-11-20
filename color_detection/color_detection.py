import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--color', default = 'yellow', type=str, help='color - yellow, purple, green')

def getContours(color_mask, img, color):
    mask = np.zeros_like(img)
    contours, hierarchy = cv2.findContours(color_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.putText(img, color, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>100:
            cv2.drawContours(img, cnt, -1,(0,0,255),3)
            cv2.drawContours(mask, cnt, -1, (0,0,255), thickness = cv2.FILLED)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.01*peri,True)
            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    return img, mask


def findColors(img,mycolors,color_name):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    color = mycolors[color_name]
    lower = np.array(color[0:3])
    upper = np.array(color[3:6])
    mask = cv2.inRange(imgHSV,lower,upper)
    return mask

if __name__ == '__main__':
    args = parser.parse_args()
    myColors = dict(yellow = [20,100,100,30,255,255], purple = [133,56,0,159,156,255],
                   blue = [57,76,0,100,255,255])
    color_name = args.color

    cap = cv2.VideoCapture(0)
    try:
        while True:
            success,img = cap.read()
            imgre = img.copy()
            color_mask = findColors(imgre,myColors,color_name)
            imgre, mask = getContours(color_mask, imgre, color_name)
            cv2.imshow("contour",imgre)
            cv2.imshow("mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        pass
