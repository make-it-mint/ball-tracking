# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import pandas as pd
import time

# video capturing from video file or camera
# to read a video file insert the file name
# for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("Test-Videos/ball_tracking_test.MP4")

# import csv compare ball tracking data
csv = pd.read_csv("X_und_Y_Positionen_des_Balles_Video_ball_tracking_test.csv")

fps = cap.get(cv.CAP_PROP_FPS)
# cap.set(cv.CAP_PROP_POS_FRAMES, fps * 57) # start video by sek 57
print(fps)

# Video soll in der richtigen Geschwindigkeit abgespielt werden / Wie viele millisekunden braucht ein einzelner Frame (querwert = Zeit in millisekunden)
frame_time = int(1000/fps)

# exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

i = 0
frame_count = 0

back_sub = cv.createBackgroundSubtractorKNN()

# stop time - comparing the times how long a algorthm takes to go through the video
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    # stop the loop when the frame is not read correctly
    if not  ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break

    fgMask = back_sub.apply(frame, learningRate = 0.7)
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # puuting the mask above the original video
    mask_frame = cv.bitwise_and(gray, gray, mask = fgMask)

    # threshhold for the ball
    ret, threshold = cv.threshold(mask_frame, 180, 255, cv.THRESH_BINARY)
    
    # searching for elipse formed shapes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9, 9))

    vid_erosion = cv.erode(threshold, kernel, iterations=2)
    vid_dilation = cv.dilate(vid_erosion, kernel, iterations=3)

 

    # finding the ball / encircle the ball 
    # find contours
    contours, _ = cv.findContours(vid_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    #detect circle
    l = []
    li = []
    
    # wie lang ist der Vektor/die Liste, um jede Stelle in der Liste/im Vektor durchlaufen zu lassen
    for k in contours:     
        x = k[:, 0][:, 0] 
        y = k[:, 0][:, 1]          
        mx = np.mean(x);         # Mittelwertbildung aller x-Koordinaten Werte 
        my = np.mean(y);         # Mittelwertbildung aller y-Koordinaten Werte, um den Mittelpunkt für den Kreis bestimmen zu können        
        #print(mx)
        #print(my)

        # circle
        r = []
        for a, b in zip(x, y):
            r.append(np.sqrt((a - mx) ** 2 + (b - my) ** 2))   # mittels des Satz des Pythagoras wird der Radius für den Kreis ermittelt
            #print(r)
        
        max(r) - min(r)

        # wenn max - min = 0 dann ist das ein perfekter Kreis                     
        li.append(k) 
        x_ball = mx
        y_ball = my

    # calculating the failsure percentage
    if len(li) == 1 and abs(x_ball - csv.x_pos[frame_count]) <= 20 and abs(y_ball - csv.y_pos[frame_count]) <= 20:
        # variabel i for right detection 
        i += 1      
        
    #print(contours)
    draw_contours = cv.drawContours(frame, li, -1, (0,0,255), 2)

    # resize
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100) # print out the width and calculate the new width
    height = int(frame.shape[0] * scale_percent / 100) # print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)

    # Display the resulting frame
    cv.imshow("frame", resized)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(frame_time) == ord("q"):
        break

    frame_count += 1

# stop time - comparing the times how long a algorthm takes to go through the video
end_time = time.time()
elasped_time = end_time - start_time
print(f"Die Ausführung des Videos hat {elasped_time}s gedauert.")

# detection of the ball percentage
detect_perc = (i * 100) / frame_count
print(f"Die Ballerkennungsrate liegt bei {detect_perc}%.")

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

