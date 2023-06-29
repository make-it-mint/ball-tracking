# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import time

import field_detection
import image_processing


# video capturing from video file or camera
# to read a video file insert the file name
# for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("Test-Videos/ball_tracking_test.mp4")

# exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()
else:
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"fps: {fps}")
    frame_time = int(1000/fps)
    # get the width and height of the video
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    # reduce the video width and heigth to match the max index
    video_height -= 1
    video_width -= 1
    print(f"video width: {video_width}")
    print(f"video height: {video_height}")

    # go to a specific frame
    #cap.set(cv.CAP_PROP_POS_FRAMES, 5210)

    ret, frame = cap.read()

    treshold = image_processing.findTreshold(image=frame)

field_found = False

x = []
y = []
 
frame_count = 0

start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    # stop the loop when the frame is not read correctly
    if not ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break

    frame_count += 1

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    _, thresh = cv.threshold(gray, treshold, 255, cv.THRESH_BINARY)

    #x, y = field_detection.findCorner(image=thresh, x_start=900, y_start=800, vertical_orientation="up", horizontal_orientation="right", video_height=video_height, video_width=video_width)
    #valid_line, x, y = field_detection.findLine(image=thresh, x=900, y=200, video_height=video_height, video_width=video_width)
    #print(valid_line)
    #upper_line, x, y = field_detection.checkFieldCenter(image=thresh, x=900, y=700, video_height=video_height, video_width=video_width)
    center_found, x, y = field_detection.findField(image=thresh, video_height=video_height, video_width=video_width)
    #field_found, x, y = field_detection.fielDetection(image=thresh, x_old=x, y_old=y, field_found=field_found, video_height=video_height, video_width=video_width)
    #print(upper_line)
    #x = [1000]
    #y = [200]

    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    for x_point, y_point in zip(x, y):
        thresh = cv.circle(thresh, (x_point,y_point), radius=3, color=(0,0,255), thickness=2)

    #thresh = cv.circle(thresh, (x,y), radius=3, color=(0,0,255), thickness=2)

    #print(thresh[0,0])

    # Display the resulting frame
    cv.namedWindow("frame", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("frame",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
    cv.imshow("frame", thresh)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(1) == ord("q"):
        break

duration = time.time() - start_time

average_fps = frame_count / duration

print(f"\nduration: \n{time.time() - start_time}s")
print(f"\naverage fps: \n{average_fps}")

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
