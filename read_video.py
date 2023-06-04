# -- coding: utf-8 --

import numpy as np
import cv2 as cv

# video capturing from video file or camera
# to read a video file insert the file name
# for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("Test-Videos/test-game.MP4")

fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_POS_FRAMES, fps * 57) # start video by sek 57
print(fps)

# Video soll in der richtigen Geschwindigkeit abgespielt werden / Wie viele millisekunden braucht ein einzelner Frame (querwert = Zeit in millisekunden)
frame_time = int(1000/fps)

# exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

# background subtraction
#backSub = cv.createBackgroundSubtractorMOG2()
backSub = cv.createBackgroundSubtractorKNN()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    # stop the loop when the frame is not read correctly
    if not  ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break

    fgMask = backSub.apply(frame, learningRate = 0.9)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # puuting the mask above the original video
    mask_frame = cv.bitwise_and(gray, gray, mask = fgMask)

    # threshhold for the ball
    ret, threshold = cv.threshold(mask_frame, 180, 255, cv.THRESH_BINARY)

    # resize
    scale_percent = 60 # percent of original size
    width = int(threshold.shape[1] * scale_percent / 100) # print out the width and calculate the new width
    height = int(threshold.shape[0] * scale_percent / 100) # print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(threshold, dim, interpolation = cv.INTER_AREA)

    # Display the resulting frame
    cv.imshow("frame", resized)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(frame_time) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
