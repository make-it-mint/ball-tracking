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

    ret, frame = cap.read()

    treshold = image_processing.findTreshold(image=frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    _, thresh = cv.threshold(gray, treshold, 255, cv.THRESH_BINARY)

    x = []
    y = []

    x_old_points = []
    y_old_points = []

    x_average = []
    y_average = []

    field_found = False

    field_detection.fielDetection(image=thresh, x_old=x, y_old=y, field_found=field_found, video_height=video_height, video_width=video_width)

    # go to a specific frame
    #cap.set(cv.CAP_PROP_POS_FRAMES, 5210)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
 
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

    """x = []
    y = []
    for _ in range(100):
        x_corner, y_corner = field_detection.findCorner(image=thresh, x_start=900, y_start=800, vertical_orientation="up", horizontal_orientation="right", video_height=video_height, video_width=video_width)
        x.append(x_corner)
        y.append(y_corner)"""
    #valid_line, x, y = field_detection.findLine(image=thresh, x=900, y=200, video_height=video_height, video_width=video_width)
    #print(valid_line)
    #upper_line, x, y = field_detection.checkFieldCenter(image=thresh, x=900, y=700, video_height=video_height, video_width=video_width)
    #center_found, x, y = field_detection.findField(image=thresh, video_height=video_height, video_width=video_width)
    field_image, field_found, field_moved, x, y = field_detection.fielDetection(image=thresh, x_old=x_average, y_old=y_average, field_found=field_found, video_height=video_height, video_width=video_width)
    #print(upper_line)
    #x = [1000]
    #y = [200]

    # save points history
    # check if the field is found and has moved
    if field_found and field_moved:
        # delete the old history
        x_old_points = []
        y_old_points = []
        
        # save the new points of the field
        x_old_points.append(x)
        y_old_points.append(y)

        # set the average variable to the new points
        x_average = x_old_points[0]
        y_average = y_old_points[0]

    # check if the field ist found and if less than a certain amount of points are saved
    elif field_found and len(x_old_points) < 10:
        # save the new points in the list
        x_old_points.append(x)
        y_old_points.append(y)

        # check if more than one set of points is saved
        if len(x_old_points) > 1:
            # calculate the average for every point
            x_average = np.mean(x_old_points, axis = 0, dtype=np.integer)
            y_average = np.mean(y_old_points, axis = 0, dtype=np.integer)

        # if only one set of points is saved
        else:
            # set the average to the one saved set
            x_average = x_old_points[0]
            y_average = y_old_points[0]

    # if the field is found
    elif field_found:
        # delete the oldest set of points
        x_old_points.pop(0)
        y_old_points.pop(0)

        # save the new set of points
        x_old_points.append(x)
        y_old_points.append(y)

        # calculate the average for every point
        x_average = np.mean(x_old_points, axis = 0, dtype=np.integer)
        y_average = np.mean(y_old_points, axis = 0, dtype=np.integer)



    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    for x_point, y_point in zip(x, y):
        thresh = cv.circle(thresh, (x_point,y_point), radius=3, color=(0,0,255), thickness=2)

    #thresh = cv.circle(thresh, (x,y), radius=3, color=(0,0,255), thickness=2)

    #print(thresh[0,0])

    # Display the resulting frame
    #cv.namedWindow("frame", cv.WND_PROP_FULLSCREEN)
    #cv.setWindowProperty("frame",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
    cv.imshow("frame", field_image)

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
