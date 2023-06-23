# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import copy
import pandas as pd
import ball_tracking_methods
import time

## a list to save the 50 first frames and to update the list for a better recognition 
frame_list = []

# kernel_size = []
# time_rate = []
# detection_rate = []

# for xk in range (3, 22, 2):
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

'''
# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        global frame, curr_x, curr_y
        frame_copy = copy.deepcopy(frame)
        cv.circle(frame_copy, (x, y), 20, (255,0,0), 2)
        global ball_pos_x, ball_pos_y
        curr_x = x
        curr_y = y 
        ## Display the resulting frame
        # cv.imshow("frame", frame_copy)
'''

## clicking the center auf the ball 
# cv.namedWindow("resized", cv.WND_PROP_FULLSCREEN)
# cv.setMouseCallback("frame",draw_circle)
# cv.setWindowProperty("resized",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)

frame_count = 0

# variabel i for right detection 
i = 0

## stop time - comparing the times how long a algorthm takes to go through the video 
start_time = time.time()

## variabel for counting frames to update the frame_list
update_frame_list = 0

while True:
    ## Capture frame-by-frame
    ret, frame = cap.read()

    ## changing the color of frame into gray tones
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ## imbinarize frame 
    _, frame_imbinarized = cv.threshold(frame_gray, 180, 255, cv.THRESH_BINARY)

    ## if loop which saves the 50 first frames in a list and updates the list ever 30 frames (1 sek)
    if len(frame_list) < 50: 
        frame_imbinarized_copy = frame_imbinarized / 255
        # frame_imbinarized_copy[np.all(frame_imbinarized_copy == (255), axis = None)] = (1)
        frame_list.append(frame_imbinarized_copy)
        background = sum(frame_list)
        background[background >= 30] = 255
        background[background < 30] = 0
        background_inverted = cv.bitwise_not(background)
    elif len(frame_list) == 50 and update_frame_list == 30:
        frame_list.pop(0)
        frame_imbinarized_copy = frame_imbinarized / 255
        # frame_imbinarized_copy[np.all(frame_imbinarized_copy == (255), axis = None)] = (1)
        frame_list.append(frame_imbinarized_copy) 
        update_frame_list = 0
        background = sum(frame_list)
        background[background >= 30] = 255
        background[background < 30] = 0
        background_inverted = cv.bitwise_not(background)
    else: 
        update_frame_list += 1

    ## if frame is read correctly ret is True
    ## stop the loop when the frame is not read correctly
    if not  ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break

    resized, contours = ball_tracking_methods.matlabDetection(frame = frame, 
                                                              frame_imbinarized = frame_imbinarized,
                                                              background_inverted = background_inverted, 
                                                              cap = cap)
    
    ## lists for the mid values
    x_mid = []
    y_mid = []


    for mid_value_contours in contours:
        # print(mid_value_contours[:, 0][:, 1])
        
        x = mid_value_contours[:, 0][:, 0]
        y = mid_value_contours[:, 0][:, 1]

        x_mid.append(round(np.mean(x)))
        y_mid.append(round(np.mean(y))) 

        # print(int(np.mean(x)))

    if len(contours) == 1:
        bp = np.array([[np.float32(x_mid[0])],[np.float32(y_mid[0])]])
        kf.correct(bp)
        tp = kf.predict()

    # calculating the falsure percentage 
    if len(contours) == 1 and abs(x_mid[0] - csv.x_pos[frame_count]) <= 20 and abs(y_mid[0] - csv.y_pos[frame_count]) <= 20:
        # variabel i for right detection 
        i += 1
    
    frame_count += 1
    

    # Display the resulting frame
    cv.imshow("resized", resized)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(1) == ord("q"):
        break

    # ball_pos_x.append(curr_x)
    # ball_pos_y.append(curr_y)

# detection of the ball percentage
detect_perc = (i * 100) / frame_count
print(f"Die Ballerkennungsrate liegt bei {detect_perc}%.")

# stop time - comparing the times how long a algorthm takes to go through the video 
end_time = time.time()
elasped_time = end_time - start_time
print(f"Die Ausführung des Videos hat {elasped_time}s gedauert.")

# kernel_size.append(xk)
# time_rate.append(elasped_time)
# detection_rate.append(detect_perc)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

## create data frame
# df = pd.DataFrame(data = {"kernel size": kernel_size, "time rate": time_rate, "detection rate": detection_rate})
## save data frame as cvs data
# df.to_csv("Daten_der unterschiedlichen_Paramter_fuer_die_Ballereknnung_Iteration_2.csv")