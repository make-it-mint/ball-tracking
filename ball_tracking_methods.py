# -- coding: utf-8 --

import numpy as np
import cv2 as cv

def thresholdBest(frame, back_sub, cap):
    
    fgMask = back_sub.apply(frame, learningRate = 0.7)
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # puuting the mask above the original video
    mask_frame = cv.bitwise_and(gray, gray, mask = fgMask)

    # threshhold for the ball
    ret, threshold = cv.threshold(mask_frame, 180, 255, cv.THRESH_BINARY)
    
    # searching for elipse formed shapes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9, 9))
    
    # mask with imrode / dilate 
    # kernel = np.ones((5, 5), np.uint8)

    vid_erosion = cv.erode(threshold, kernel, iterations=2)
    vid_dilation = cv.dilate(vid_erosion, kernel, iterations=3)

    vid_rgb = cv.cvtColor(vid_dilation, cv.COLOR_GRAY2RGB)

    # finding the ball / encircle the ball 
    # find contours
    contours, _ = cv.findContours(vid_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw 
    vid_draw = cv.drawContours(frame, contours, -1, (0, 0, 255), 3)

    # resize
    scale_percent = 60 # percent of original size
    width = int(vid_draw.shape[1] * scale_percent / 100) # print out the width and calculate the new width
    height = int(vid_draw.shape[0] * scale_percent / 100) # print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(vid_draw, dim, interpolation = cv.INTER_AREA)

    cv.rectangle(resized, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(resized, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    return resized, contours

def kalman_method():
    kalman_fil = cv.KalmanFilter(4,2)
    kalman_fil.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman_fil.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman_fil.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 1

    return kalman_fil

def thresholdChangedOrder(frame, back_sub, cap):
      
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # threshhold for the ball
    ret, threshold = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)

    fgMask = back_sub.apply(threshold, learningRate = 0.7)

    # puuting the mask above the original video
    mask_frame = cv.bitwise_and(threshold, threshold, mask = fgMask)
    
    # searching for elipse formed shapes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9, 9))
    
    # mask with imrode / dilate 
    # kernel = np.ones((5, 5), np.uint8)

    vid_erosion = cv.erode(mask_frame, kernel, iterations=2)
    vid_dilation = cv.dilate(vid_erosion, kernel, iterations=3)

    vid_rgb = cv.cvtColor(vid_dilation, cv.COLOR_GRAY2RGB)

    # finding the ball / encircle the ball 
    # find contours
    contours, _ = cv.findContours(vid_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw 
    vid_draw = cv.drawContours(frame, contours, -1, (0, 0, 255), 3)

    # resize
    scale_percent = 60 # percent of original size
    width = int(vid_draw.shape[1] * scale_percent / 100) # print out the width and calculate the new width
    height = int(vid_draw.shape[0] * scale_percent / 100) # print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(vid_draw, dim, interpolation = cv.INTER_AREA)

    cv.rectangle(resized, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(resized, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    return resized, contours

def threshold(frame, back_sub, cap):
    
    fgMask = back_sub.apply(frame, learningRate = 0.2)
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # puuting the mask above the original video
    mask_frame = cv.bitwise_and(gray, gray, mask = fgMask)

    # threshhold for the ball
    ret, threshold = cv.threshold(mask_frame, 180, 255, cv.THRESH_BINARY)
    
    # searching for elipse formed shapes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))
    
    # mask with imrode / dilate 
    # kernel = np.ones((5, 5), np.uint8)

    vid_erosion = cv.erode(threshold, kernel, iterations=1)
    vid_dilation = cv.dilate(vid_erosion, kernel, iterations=3)

    vid_rgb = cv.cvtColor(vid_dilation, cv.COLOR_GRAY2RGB)

    # finding the ball / encircle the ball 
    # find contours
    contours, _ = cv.findContours(vid_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw 
    vid_draw = cv.drawContours(frame, contours, -1, (0, 0, 255), 3)

    # resize
    scale_percent = 60 # percent of original size
    width = int(vid_draw.shape[1] * scale_percent / 100) # print out the width and calculate the new width
    height = int(vid_draw.shape[0] * scale_percent / 100) # print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(vid_draw, dim, interpolation = cv.INTER_AREA)

    cv.rectangle(resized, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(resized, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    return resized, contours

def matlabDetection(frame, frame_imbinarized, background_inverted, x_ball_fr_mid, y_ball_fr_mid, x_left, x_right, y_upper, y_lower):

    ## combinig the imbinarized frame with the inverted one
    first_frame_combined = cv.bitwise_and(frame_imbinarized, background_inverted)
    first_frame_combined_cor = first_frame_combined[y_upper:y_lower, x_left:x_right] 

    ## searching for elipse formed shapes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))

    vid_erosion = cv.erode(first_frame_combined_cor, kernel, iterations=1)
    vid_dilation = cv.dilate(vid_erosion, kernel, iterations=2)

    ## finding the ball / encircle the ball 
    ## find contours
    contours, _ = cv.findContours(vid_dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE, offset = (x_left, y_upper))

    con_filtered = []

    ## filtering to big detected contours 
    if len(contours) > 0: 
        for con in range(0, len(contours)):
            con_area_list = cv.contourArea(contours[con])
            if con_area_list < 3500:
                con_filtered.append(contours[con])
    else:
        con_filtered = contours

    ## lists for the mid values
    x_mid = []
    y_mid = []

    circle_contour_list = []
    total_distance = []

    for mid_value_contours in con_filtered:
        
        x = mid_value_contours[:, 0][:, 0]
        y = mid_value_contours[:, 0][:, 1]

        x_mid.append(round(np.mean(x)))
        y_mid.append(round(np.mean(y)))

        if len(x_mid) >= 1 and len(y_mid) >= 1:
            x_distance = abs(x_mid[-1] - x_ball_fr_mid) 
            y_distance = abs(y_mid[-1] - y_ball_fr_mid)
            total_distance.append(np.sqrt(x_distance ** 2 + y_distance ** 2))

    if len(total_distance) > 1:
        ball = con_filtered[total_distance.index(min(total_distance))]
        x_ball = [x_mid[total_distance.index(min(total_distance))]]
        y_ball = [y_mid[total_distance.index(min(total_distance))]]

    else:
        ball = con_filtered
        x_ball = x_mid
        y_ball = y_mid

    # if x_mid and x_mid_old == 1 and y_mid and y_mid_old == 1:
        # x_vel = x_mid - x_mid_old
        # y_vel = y_mid - y_mid_old

    vid_draw_0 = cv.drawContours(frame, con_filtered, -1, (255, 0, 0 ), 3)
    ## draw 
    vid_draw = cv.drawContours(frame, ball, -1, (0, 0, 255), 3)

    ## resize
    scale_percent = 60 ## percent of original size
    width = int(vid_draw.shape[1] * scale_percent / 100) ## print out the width and calculate the new width
    height = int(vid_draw.shape[0] * scale_percent / 100) ## print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(vid_draw, dim, interpolation = cv.INTER_AREA)

    return resized, con_filtered, ball, x_ball, y_ball, x_mid, y_mid # x_vel, y_vel

def iTriedButIFailedAndOtherStuffITried():
    '''
    ## defining the form of the ball 
    ## circle
    r = []
    for a, b in zip(x, y):
        ## with pytagoras defining the radius of the ball/circle 
        r.append(np.sqrt((a - x_mid[-1]) ** 2 + (b - y_mid[-1]) ** 2))  
    max(r) - min(r)

    ## if max - min = 0 it means a oerfect circle
    if np.max(r) - np.min(r) < 15 and np.max(r) < 30:
        circle_contour_list.append(mid_value_contours)
    else: 
        x_mid.pop(-1)
        y_mid.pop(-1)
    '''

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
    