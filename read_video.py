# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import copy
import pandas as pd
import ball_tracking_methods
import time

## a list to save the 50 first frames and to update the list for a better recognition 
frame_list = []

kernel_size = []
time_rate = []
detection_rate = []
iterations = []
update_rate = []
pixel_detection = []

# for pxc in range(5, 16, 2):
    # for upd in range(4, 30, 5):
        # for iter in range(1, 3):
            # for xk in range (7, 20, 2):
## video capturing from video file or camera
## to read a video file insert the file name
## for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("Test-Videos/ball_tracking_test.MP4")

# import csv compare ball tracking data
csv = pd.read_csv("X_und_Y_Positionen_des_Balles_Video_ball_tracking_test.csv")
print(csv.x_pos[0])

fps = cap.get(cv.CAP_PROP_FPS)
# cap.set(cv.CAP_PROP_POS_FRAMES, fps * 57) # start video by sek 57
print(fps)

## Video soll in der richtigen Geschwindigkeit abgespielt werden / Wie viele millisekunden braucht ein einzelner Frame (querwert = Zeit in millisekunden)
frame_time = int(1000/fps)

## start of manual background subtraction for a better motion detection 
## memorizing the first frame of the video
ret, first_frame = cap.read()
## changing the color of the frist frame into gray tones
first_frame_gray_vid = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
## changing the color into back and white
_, first_frame_imbinarized = cv.threshold(first_frame_gray_vid, 180, 255, cv.THRESH_BINARY)
## imcomplement the first frame 
first_frame_imbinarized_inverted = cv.bitwise_not(first_frame_imbinarized)

## exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

## background subtraction
# back_sub = cv.createBackgroundSubtractorMOG2()
back_sub = cv.createBackgroundSubtractorKNN()

# implementin kalman function
kf = ball_tracking_methods.kalman_method()

ball_pos_x = []
ball_pos_y = []
curr_x = 0
curr_y = 0

## searching for the ball and rembering its last position
x_ball_found_remember = []
y_ball_found_remember = []

x_ball_fr_mid = 960 
y_ball_fr_mid = 540

## clicking the center auf the ball 
cv.namedWindow("resized", cv.WND_PROP_FULLSCREEN)
# cv.setMouseCallback("frame",draw_circle)
cv.setWindowProperty("resized",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)

frame_count = 0

## variabel i for right detection 
i = 0

## variabel j for right detection kalman 
j = 0

## stop time - comparing the times how long a algorthm takes to go through the video 
start_time = time.time()

## variabel for counting frames to update the frame_list
update_frame_list = 0

x_mid_old = []
y_mid_old = []
x_vel = 0
y_vel = 0

kf_predict = np.array([[np.float32(0)],[np.float32(0)]])

while True:
    ## Capture frame-by-frame
    ret, frame = cap.read()

    ## if frame is read correctly ret is True
    ## stop the loop when the frame is not read correctly
    if not  ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break

    ## changing the color of frame into gray tones
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ## imbinarize frame 
    _, frame_imbinarized = cv.threshold(frame_gray, 180, 1, cv.THRESH_BINARY)

    ## if loop which saves the 50 first frames in a list and updates the list ever 30 frames (1 sek)
    if len(frame_list) < 50: 
        frame_imbinarized_copy = frame_imbinarized
        # frame_imbinarized_copy[np.all(frame_imbinarized_copy == (255), axis = None)] = (1)
        while len(frame_list) < 50:
            frame_list.append(frame_imbinarized_copy)
        background = sum(frame_list)
        background[background >= 6] = 255
        background[background < 6] = 0
        background_inverted = cv.bitwise_not(background)
    elif len(frame_list) == 50 and update_frame_list == 5:
        frame_list.pop(0)
        frame_imbinarized_copy = frame_imbinarized
        # frame_imbinarized_copy[np.all(frame_imbinarized_copy == (255), axis = None)] = (1)
        frame_list.append(frame_imbinarized_copy) 
        update_frame_list = 0
        background = sum(frame_list)
        background[background >= 6] = 255
        background[background < 6] = 0
        background_inverted = cv.bitwise_not(background)
    else: 
        update_frame_list += 1

    resized, contours, ball, x_ball, y_ball, x_mid, y_mid = ball_tracking_methods.matlabDetection(
        frame = frame, 
        frame_imbinarized = frame_imbinarized,
        background_inverted = background_inverted, 
        cap = cap,
        x_ball_fr_mid = (kf_predict[0] + x_ball_fr_mid) / 2,
        y_ball_fr_mid = (kf_predict[1] + y_ball_fr_mid) / 2)
    
    ## searching for the ball and rembering its last position
    if len(x_ball_found_remember) and len(y_ball_found_remember) < 5 and len(ball) == 1:
        x_ball_found_remember.append(x_ball)
        y_ball_found_remember.append(y_ball)
        x_ball_fr_mid = round(np.mean(x_ball_found_remember))
        y_ball_fr_mid = round(np.mean(y_ball_found_remember))
    elif len(x_ball_found_remember) and len(y_ball_found_remember) > 5 and len(ball) == 1:
        x_ball_found_remember.pop(-1)
        y_ball_found_remember.pop(-1)
        x_ball_found_remember.append(x_ball)
        y_ball_found_remember.append(y_ball)
        x_ball_fr_mid = round(np.mean(x_ball_found_remember))
        y_ball_fr_mid = round(np.mean(y_ball_found_remember))
    elif len(ball) == 1:
        x_ball_found_remember.append(x_ball)
        y_ball_found_remember.append(y_ball)
        x_ball_fr_mid = round(np.mean(x_ball_found_remember))
        y_ball_fr_mid = round(np.mean(y_ball_found_remember))

    ## kalman filter
    # kf.statePre = np.array([[csv.x_pos[frame_count]], [csv.y_pos[frame_count]], [0], [0]], np.float32)
    # kf.statePost = np.array([[csv.x_pos[frame_count]], [csv.y_pos[frame_count]], [0], [0]], np.float32)
    if len(x_ball) and len(y_ball) == 1:
        # kf.statePre = np.array([[x_mid[0]], [y_mid[0]], [x_vel], [y_vel]], np.float32)
        # kf.statePost = np.array([[x_mid[0]], [y_mid[0]], [x_vel], [y_vel]], np.float32)
        kf.correct(np.array([[np.float32(x_ball[0])],[np.float32(y_ball[0])]]))
    # else:
        # kf_predict = kf.predict()

    # kf.correct(np.array([[np.float32(x_mid[0])],[np.float32(y_mid[0])]]))
    # kf_predict = kf.predict()
    cv.circle(frame, (int(kf_predict[0, 0]), int(kf_predict[1, 0])), 20, (0,255,0), 2)

    ## calculating the failsure percentage 
    if len(ball) == 1 and abs(x_mid[0] - csv.x_pos[frame_count]) <= 20 and abs(y_mid[0] - csv.y_pos[frame_count]) <= 20:
        # variabel i for right detection 
        i += 1
    elif abs(kf_predict[0, 0] - csv.x_pos[frame_count]) <= 20 and abs(kf_predict[1, 0] - csv.y_pos[frame_count]) <= 20:
        i += 1
    
    ## calculating the failsure percentage for kalman
    if abs(kf_predict[0, 0] - csv.x_pos[frame_count]) <= 20 and abs(kf_predict[1, 0] - csv.y_pos[frame_count]) <= 20:
        # variabel i for right detection 
        j += 1

    frame_count += 1 

    kf_predict = kf.predict()
    # cv.circle(frame, (int(kf_predict[0, 0]), int(kf_predict[1, 0])), 20, (255,0,0), 2)
    
    ## Display the resulting frame
    cv.imshow("resized", frame)

    ## stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(1) == ord("q"):
        break

    x_mid_old = x_ball
    y_mid_old = y_ball 

# detection of the ball percentage
detect_perc = (i * 100) / frame_count
print(f"Die Ballerkennungsrate liegt bei {detect_perc}%.")

# detection of the ball percentage
detect_perc_kalman = (j * 100) / frame_count
print(f"Die Ballerkennungsrate mit dem Klaman Filter liegt bei {detect_perc_kalman}%.")

# stop time - comparing the times how long a algorthm takes to go through the video 
end_time = time.time()
elasped_time = end_time - start_time
print(f"Die Ausführung des Videos hat {elasped_time}s gedauert.")

# kernel_size.append(xk)
# time_rate.append(elasped_time)
# detection_rate.append(detect_perc)
# iterations.append(iter)
# update_rate.append(upd)
# pixel_detection.append(pxc)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

## create data frame
# df = pd.DataFrame(data = {"pixel count":pixel_detection, "update rate": update_rate, "iterations": iterations, 
            # "kernel size": kernel_size, "time rate": time_rate, "detection rate": detection_rate})
## save data frame as cvs data
# df.to_csv("Erkennungsdaten_eigene_backgroundsubstraction_2.csv")