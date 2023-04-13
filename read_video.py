# -- coding: utf-8 --

import numpy as np
import cv2 as cv

# video capturing from video file or camera
# to read a video file insert the file name
# for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("videos/GX010003.MP4")

fps = cap.get(cv.CAP_PROP_FPS)
print(fps)
frame_time = int(1000/fps)

# exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    # stop the loop when the frame is not read correctly
    if not  ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow("frame", gray)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(frame_time) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()