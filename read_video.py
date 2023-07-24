# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import copy
import mysql.connector
import dotenv
import os
import datetime
import multiprocessing
import pandas as pd
import ball_tracking_methods
import time

import field_detection
import image_processing

def save_data_process(queue_in):
    """
    function to load the data into the database.
    """

    # load the .env file wich contains the username and password for the MySQL connection
    dotenv.load_dotenv()

    # open a connection to the MySQL datbase
    mydb = mysql.connector.connect(
        host="192.168.116.52",
        user=str(os.getenv("USER")),
        password=str(os.getenv("PASSWORD"))
    )

    # create a curser object to interact with the database
    mycurser = mydb.cursor()

    # create the database if it doesn't exist
    mycurser.execute("CREATE DATABASE IF NOT EXISTS ball_tracking")

    # show all the Databases
    mycurser.execute("SHOW DATABASES")

    # print the databases
    print("Databases")
    for x in mycurser:
        print(x)
    print("")

    # swwitch to the ball_tracking database
    mycurser.execute("USE ball_tracking")

    #mycurser.execute("DROP TABLE IF EXISTS positions")
    #mycurser.execute("DROP TABLE IF EXISTS sessions")
    #mycurser.execute("DROP TABLE IF EXISTS videos")

    # create a table for video informations if it doesn't exist
    mycurser.execute(
        "CREATE TABLE IF NOT EXISTS videos( \
        video_id INT NOT NULL AUTO_INCREMENT, \
        video_name VARCHAR(500) NOT NULL, \
        fps DOUBLE NOT NULL, \
        frames INT NOT NULL, \
        duration DOUBLE NOT NULL, \
        PRIMARY KEY (video_id), \
        UNIQUE(video_name) \
        )"
    )

    # create a table for session informations if it doesn't exist
    mycurser.execute(
        "CREATE TABLE IF NOT EXISTS sessions( \
        session_id INT NOT NULL AUTO_INCREMENT, \
        start DATETIME(0) NOT NULL, \
        end DATETIME(0), \
        video_id INT, \
        PRIMARY KEY (session_id), \
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON UPDATE CASCADE ON DELETE RESTRICT \
        )"
    )

    # create a table for the positions if the table doesn't exist
    mycurser.execute(
        "CREATE TABLE IF NOT EXISTS positions( \
        session_id INT, \
        position_id INT, \
        ball_x INT, \
        ball_y INT, \
        kalman_x INT, \
        kalman_y INT, \
        field_center_x INT, \
        field_center_y INT, \
        time DATETIME(6), \
        PRIMARY KEY (session_id, position_id), \
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON UPDATE CASCADE ON DELETE CASCADE \
        )"
    )

    # show all the tables
    mycurser.execute("SHOW TABLES")

    # print the tables
    print("Tables in ball_tracking")
    for x in mycurser:
        print(x)
    print("")

    # get the video file informations from the queue
    while True:
        video_informations = queue_in.get()
        if video_informations is not None:
            break
        elif video_informations == "STOP":
            # close the MySQL curser object
            mycurser.close()
            # close the MySQL connection
            mydb.close()
            return


    # check if the video is already in the database
    mycurser.execute("SELECT COUNT(*) FROM videos WHERE video_name = %s", (video_informations[0],))
    if mycurser.fetchall()[0][0] == 0:
        video_exist = False
    else:
        video_exist = True

    print(f"video exists: {video_exist}")

    # insert the video in the Database if it doesn't exist
    if not video_exist:
        sql = "INSERT INTO videos (video_name, fps, frames, duration) VALUES (%s, %s, %s, %s)"
        #val = (video_file, fps, total_frames, video_duration)
        mycurser.execute(sql, video_informations)

    # get the video id from the database
    mycurser.execute("SELECT video_id FROM videos WHERE video_name = %s", (video_informations[0],))
    video_id = mycurser.fetchall()[0][0]
    print(f"video id: {video_id}")

    # get the start_time from the queue
    while True:
        start_time = queue_in.get()
        if start_time is not None:
            break

    sql = "INSERT INTO sessions (start, video_id) VALUES (%s, %s)"
    val = (start_time, video_id)
    mycurser.execute(sql, val)

    mydb.commit()

    mycurser.execute("SELECT last_insert_id()")

    session_id = mycurser.fetchall()[0][0]


    while True:
        # get the points data from the queue
        while True:
            data = queue_in.get()
            if data is not None:
                break
        
        # stop the loop if a certain string is send 
        if data == "STOP":
            break

        # convert the data into the needed format
        val = []
        for frame_data in data:
            val.append((session_id, frame_data[0], frame_data[1], frame_data[2], frame_data[3]))

        # save the data in the database
        sql = "INSERT INTO positions (session_id, position_id, field_center_x, field_center_y, time) VALUES (%s, %s, %s, %s, %s)"
        mycurser.executemany(sql, val)

        mydb.commit()

    # get the end_time from the queue
    while True:
        end_time = queue_in.get()
        if end_time is not None:
            break

    # save the end time in the database
    sql = "UPDATE sessions SET end = (%s) WHERE session_id = (%s)"
    val = (end_time, session_id)
    mycurser.execute(sql, val)

    mydb.commit()

    # print the video table
    mycurser.execute("SELECT * FROM videos")
    results = mycurser.fetchall()
    print("videos:")
    for x in results:
        print(x)
    print("")

    # print the sessions table
    mycurser.execute("SELECT * FROM sessions")
    results = mycurser.fetchall()
    print("sessions:")
    for x in results:
        print(x)
    print("")
    """
    # print the positions table
    mycurser.execute("SELECT * FROM positions")
    results = mycurser.fetchall()
    print("positions:")
    for x in results:
        print(x)
    print("")
    """
    # close the MySQL curser object
    mycurser.close()
    # close the MySQL connection
    mydb.close()

def field_detection_process(queue_out):

    video_file = "ball_tracking_test.mp4"

    # video capturing from video file or camera
    # to read a video file insert the file name
    # for a camera insert an integer depending on the camera port
    cap = cv.VideoCapture("Test-Videos/" + video_file)

    # exit the programm if the camera cannot be oppend, or the video file cannot be read
    if not cap.isOpened():
        print("Cannot open camera or video file")
        # put a STOP string into the queue
        queue.put("STOP")
        return

    # get the video fps
    fps = cap.get(cv.CAP_PROP_FPS)
    # get the total frame count of the video
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    # calculate the duration of the video
    video_duration = total_frames / fps

    # put the video informations in the queue
    queue.put((video_file, fps, total_frames, video_duration))

    # print the fps
    print(f"fps: {fps}")
    # calculate the frame time 
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

    # list for the database
    db_session_id = []
    db_frame_count = []
    db_field_center_x = []
    db_field_center_y = []
    db_time = []
    data = []

    start_time = datetime.datetime.now()

    # put the start time into the queue
    queue.put(start_time)

    print(f"start time: {start_time}")

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
        field_image, field_found, field_moved, x, y = field_detection.fielDetection(
            image=thresh, x_old=x_average, y_old=y_average, field_found=field_found, video_height=video_height, video_width=video_width
            )
        #print(upper_line)
        #x = [1000]
        #y = [200]

        if field_found and field_moved:

            dx1 = abs(x[14] - x[15])
            dy1 = abs(y[14] - y[15])

            dx2 = abs(x[19] - x[14])
            dy2 = abs(y[19] - y[14])

            dx3 = abs(x[20] - x[14])
            dy3 = abs(y[20] - y[14])

            dx4 = abs(x[14] - x[16])
            dy4 = abs(y[14] - y[16])

            ipt = 0.78

            rect = np.array([
                [x[15]-dx1*ipt, y[15]-dy1*ipt],
                [x[19]+dx2*ipt, y[19]-dy2*ipt],
                [x[20]+dx3*ipt, y[20]+dy3*ipt],
                [x[16]-dx4*ipt, y[16]+dy4*ipt]],
                dtype="float32"
                )

            dst = np.array([
                [0,0],
                [video_width, 0],
                [video_width, video_height],
                [0, video_height]],
                dtype="float32"
            )

            M = cv.getPerspectiveTransform(rect, dst)
            warped = cv.warpPerspective(frame, M, (video_width, video_height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        elif field_found:
            warped = cv.warpPerspective(frame, M, (video_width, video_height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        else:
            warped = thresh
        #warped = frame

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
        cv.namedWindow("frame", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("frame",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
        cv.imshow("frame", warped)

        if field_found:
            """db_session_id.append(session_id)
            db_frame_count.append(frame_count)
            db_field_center_x.append(x[14])
            db_field_center_y.append(y[14])
            db_time.append(datetime.datetime.now())"""

            # save the data into a list
            data.append((frame_count, x[14], y[14], datetime.datetime.now()))

            # save the end time in the database
            """
            sql = "INSERT INTO positions (session_id, position_id, field_center_x, field_center_y, time) VALUES (%s, %s, %s, %s, %s)"
            val = (session_id, frame_count, x[14], y[14], datetime.datetime.now())
            mycurser.execute(sql, val)

            mydb.commit()
            """
            if len(data) >= 500:
                # put the point data into the queue
                queue.put(data)
                # clear the data list
                data = []

        # stop the loop if the "q" key on the keyboard is pressed 
        if cv.waitKey(1) == ord("q"):
            break
    
    # put the point data into the queue
    queue.put(data)
    # put a STOP string into the queue
    queue.put("STOP")

    end_time = datetime.datetime.now()

    # put the end time into the queue
    queue.put(end_time)

    duration = end_time - start_time
    duration = duration.total_seconds()

    average_fps = frame_count / duration

    print(f"\nduration: \n{duration}s")
    print(f"\naverage fps: \n{average_fps}")

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

def ball_tracking_process(queue_in, queue_out):
    """
    function for the ball detection process
    """

    ## a list to save the 50 first frames and to update the list for a better recognition 
    frame_list = []

    kernel_size = []
    time_rate = []
    detection_rate = []
    iterations = []
    update_rate = []
    pixel_detection = []

    # for _ in range(3):
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
    else:
        _, frame_ip = cap.read()
        threshold_ip = image_processing.findTreshold(image = frame_ip)

        ## get the width and height of the video
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))
        ## reduce the video width and heigth to match the max index
        video_height -= 1
        video_width -= 1

        cap.set(cv.CAP_PROP_POS_FRAMES, 0)


    x = []
    y = []

    x_old_points = []
    y_old_points = []

    x_average = []
    y_average = []

    field_found = False


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
    kf_pred_old = np.array([[np.float32(0)],[np.float32(0)]])

    kf_update = 10

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))

    while True:
        ## Capture frame-by-frame
        ret, frame = cap.read()

        ## if frame is read correctly ret is True
        ## stop the loop when the frame is not read correctly
        if not  ret:
            print("Can't recive frame (stream end?). Exiting ...")
            break

        field_image, field_found, field_moved, x, y, x_left, x_right, y_lower, y_upper = field_detection.fieldDetection(
            image_color = frame, 
            x_old = x_average,
            y_old = y_average,
            field_found = field_found,
            video_height = video_height,
            video_width = video_width,
            threshold = threshold_ip
        )

        ## save points history
        ## check if the field is found and has moved
        if field_found and field_moved:
            # delete the old history
            x_old_points = []
            y_old_points = []

            ## save the new points of the field
            x_old_points.append(x)
            y_old_points.append(y)

            ## set the average variable to the new points
            x_average = x_old_points[0]
            y_average = y_old_points[0]

        ## check if the field ist found and if less than a certain amount of points are saved
        elif field_found and len(x_old_points) < 10:
            # save the new points in the list
            x_old_points.append(x)
            y_old_points.append(y)

            ## check if more than one set of points is saved
            if len(x_old_points) > 1:
                # calculate the average for every point
                x_average = np.mean(x_old_points, axis = 0, dtype=np.integer)
                y_average = np.mean(y_old_points, axis = 0, dtype=np.integer)

            ## if only one set of points is saved
            else:
                # set the average to the one saved set
                x_average = x_old_points[0]
                y_average = y_old_points[0]

        ## if the field is found
        elif field_found:
            ## delete the oldest set of points
            x_old_points.pop(0)
            y_old_points.pop(0)

            ## save the new set of points
            x_old_points.append(x)
            y_old_points.append(y)

            ## calculate the average for every point
            x_average = np.mean(x_old_points, axis = 0, dtype=np.integer)
            y_average = np.mean(y_old_points, axis = 0, dtype=np.integer)

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
            background[background >= 3] = 255
            background[background < 3] = 0
            background_inverted = cv.bitwise_not(background)
        elif len(frame_list) == 50 and update_frame_list == 9:
            frame_list.pop(0)
            frame_imbinarized_copy = frame_imbinarized
            # frame_imbinarized_copy[np.all(frame_imbinarized_copy == (255), axis = None)] = (1)
            frame_list.append(frame_imbinarized_copy) 
            update_frame_list = 0
            background = sum(frame_list)
            background[background >= 3] = 255
            background[background < 3] = 0
            background_inverted = cv.bitwise_not(background)
        else: 
            update_frame_list += 1

        resized, contours, ball, x_ball, y_ball, x_mid, y_mid = ball_tracking_methods.matlabDetection(
            frame = frame, 
            frame_imbinarized = frame_imbinarized,
            background_inverted = background_inverted, 
            cap = cap,
            x_ball_fr_mid = (kf_predict[0, 0] + x_ball_fr_mid) / 2,
            y_ball_fr_mid = (kf_predict[1, 0] + y_ball_fr_mid) / 2,
            x_right = x_right,
            x_left = x_left,
            y_upper = y_upper,
            y_lower = y_lower
            )
        
        cv.circle(frame, (int(x_ball_fr_mid), int(y_ball_fr_mid)), 20, (20,125,35), 2)
        
        ## searching for the ball and rembering its last position
        if len(x_ball_found_remember) and len(y_ball_found_remember) < 10 and len(ball) == 1:
            x_ball_found_remember.append(x_ball)
            y_ball_found_remember.append(y_ball)
            x_ball_fr_mid = round(np.mean(x_ball_found_remember))
            y_ball_fr_mid = round(np.mean(y_ball_found_remember))
        elif len(x_ball_found_remember) and len(y_ball_found_remember) > 10 and len(ball) == 1:
            x_ball_found_remember.pop(0)
            y_ball_found_remember.pop(0)
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
        cv.circle(frame, (int(csv.x_pos[frame_count]), int(csv.y_pos[frame_count])), 20, (127,0,125), 2)

        ## calculating the failsure percentage 
        if len(ball) == 1 and abs(x_ball - csv.x_pos[frame_count]) <= 20 and abs(y_ball - csv.y_pos[frame_count]) <= 20:
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

        ball_radius = 31
        kalman_corrected = False 

        ## barrier for the kalaman 
        if kf_predict[0, 0] < x_left + ball_radius:
            kf_distance_y = kf_pred_old[1, 0] - kf_predict[1, 0]
            # print(f"old kf predict is: {kf_predict}")
            new_kf_predict_x_left = abs(x_left + ball_radius - kf_predict[0, 0])
            old_kf_predict_x_left = abs(x_left + ball_radius - kf_pred_old[0, 0])
            kf_propotion = new_kf_predict_x_left / (new_kf_predict_x_left + old_kf_predict_x_left)
            kf_new_point = kf_distance_y * kf_propotion + kf_predict[1, 0]
            kf.correct(np.array([[np.float32(x_left + ball_radius)],[np.float32(kf_new_point)]]))  
            kf_predict[0, 0] = x_left + ball_radius + new_kf_predict_x_left
            kalman_corrected = True
        elif kf_predict[0, 0] > x_right - ball_radius:
            kf_distance_y = kf_pred_old[1, 0] - kf_predict[1, 0]
            new_kf_predict_x_right = abs(x_right - ball_radius - kf_predict[0, 0])
            old_kf_predict_x_right = abs(x_right - ball_radius - kf_pred_old[0, 0])
            kf_propotion = new_kf_predict_x_right / (new_kf_predict_x_right + old_kf_predict_x_right)
            kf_new_point = kf_distance_y * kf_propotion + kf_predict[1, 0]
            kf.correct(np.array([[np.float32(x_right - ball_radius)],[np.float32(kf_new_point)]]))
            kf_predict[0, 0] = x_right - ball_radius - new_kf_predict_x_right
            kalman_corrected = True

        if kf_predict[1, 0] < y_upper + ball_radius:
            kf_distance_x = kf_pred_old[0, 0] - kf_predict[0, 0]
            new_kf_predict_y_upper = abs(y_upper + ball_radius - kf_predict[1, 0])
            old_kf_predict_y_upper = abs(y_upper + ball_radius - kf_pred_old[1, 0])
            kf_propotion = new_kf_predict_y_upper / (new_kf_predict_y_upper + old_kf_predict_y_upper)
            kf_new_point = kf_distance_x * kf_propotion + kf_predict[0, 0]
            kf.correct(np.array([[np.float32(y_upper + ball_radius)],[np.float32(kf_new_point)]]))
            kf_predict[1, 0] = y_upper + ball_radius + new_kf_predict_y_upper
            kalman_corrected = True
        elif kf_predict[1, 0] > y_lower - ball_radius:
            kf_distance_x = kf_pred_old[0, 0] - kf_predict[0, 0]
            new_kf_predict_y_lower = abs(y_lower - ball_radius - kf_predict[1, 0])
            old_kf_predict_y_lower = abs(y_lower - ball_radius - kf_pred_old[1, 0])
            kf_propotion = new_kf_predict_y_lower / (new_kf_predict_y_lower + old_kf_predict_y_lower)
            kf_new_point = kf_distance_x * kf_propotion + kf_predict[1, 0]
            kf.correct(np.array([[np.float32(y_lower - ball_radius)],[np.float32(kf_new_point)]]))
            kf_predict[1, 0] = y_lower - ball_radius - new_kf_predict_y_lower
            kalman_corrected = True
        
        if kalman_corrected:
            kf.correct(np.array([[np.float32(kf_predict[0, 0])],[np.float32(kf_predict[1, 0])]])) 

        if np.sqrt((kf_predict[0, 0] - kf_pred_old[0, 0]) ** 2 + (kf_predict[1, 0]- kf_pred_old[1, 0]) ** 2) > 150 and kf_update < 10:
            kf_update += 1
            kf_predict = kf_pred_old
        else:
            kf_update = 0 

        kf_pred_old = kf_predict  

        out.write(frame)     

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

    ## When everything done, release the capture
    cap.release()
    out.release()
    cv.destroyAllWindows()

    ## create data frame
    # df = pd.DataFrame(data = {"pixel count":pixel_detection, "update rate": update_rate, "iterations": iterations, 
                # "kernel size": kernel_size, "time rate": time_rate, "detection rate": detection_rate})
    ## save data frame as cvs data
    # df.to_csv("Erkennungsdaten_eigene_backgroundsubstraction_2.csv")


if __name__ == "__main__":
    # create the queue to send data from one process to the other
    queue = multiprocessing.Queue()

    # create the procces to save the data into the database and start it
    db_process = multiprocessing.Process(target=save_data_process, args=(queue,))
    db_process.start()

    # create the ball tracking process and start it
    bt_process = multiprocessing.Process(target=field_detection_process, args=(queue,))
    bt_process.start()

    # wait for the processes to finish
    db_process.join()
    bt_process.join()