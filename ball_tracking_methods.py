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

def matlabDetection(frame, frame_imbinarized, background_inverted, cap):

    #print(type(frame_imbinarized[0][0]))
    #print(frame_imbinarized[0][0])
    #print(type(background_inverted[0][0]))
    #print(background_inverted[0][0])

    ## combinig the imbinarized frame with the inverted one
    first_frame_combined = cv.bitwise_and(frame_imbinarized, background_inverted) 

    ## searching for elipse formed shapes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))

    vid_erosion = cv.erode(first_frame_combined, kernel, iterations=1)
    vid_dilation = cv.dilate(vid_erosion, kernel, iterations=3)

    ## finding the ball / encircle the ball 
    ## find contours
    contours, _ = cv.findContours(vid_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ## draw 
    vid_draw = cv.drawContours(frame, contours, -1, (0, 0, 255), 3)

    ## resize
    scale_percent = 60 ## percent of original size
    width = int(vid_draw.shape[1] * scale_percent / 100) ## print out the width and calculate the new width
    height = int(vid_draw.shape[0] * scale_percent / 100) ## print out the height and calculate the new height
    dim = (width, height)

    resized = cv.resize(vid_draw, dim, interpolation = cv.INTER_AREA)

    cv.rectangle(resized, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(resized, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    return resized, contours