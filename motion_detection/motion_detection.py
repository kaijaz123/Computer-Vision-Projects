import cv2
import imutils

def motion_detection():
    video_capture = cv2.VideoCapture(0)
    first_frame = None
    while True:
        frame = video_capture.read()[1]
        text = 'Unoccupied'

        grayscale_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(grayscale_frame,(21,21),0)
        blur_frame = cv2.blur(gaussian_frame,(5,5))

        if first_frame is None:
            first_frame = grayscale_frame
        else:
            pass
        frame = imutils.resize(frame,width=500)
        frame_delta = cv2.absdiff(first_frame,grayscale_frame)
        thresh = cv2.threshold(frame_delta,50,255,cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh,None,iterations=2)
        contours, _ = cv2.findContours(dilate_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for index,contour in enumerate(contours):
            if cv2.contourArea(contour) > 800:
                (x,y,w,h) = cv2.boundingRect(contour)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0) ,2)
                text = 'Occupied'
            else:
                pass
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,f'[+] Room Status:{text}',(10,20),font,0.5,(0,0,225),2)
        cv2.imshow('Frame',frame)
        cv2.imshow('Threshold',thresh)
        cv2.imshow('Frame_delta',frame_delta)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    motion_detection()
