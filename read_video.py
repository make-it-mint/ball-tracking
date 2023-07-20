# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import time
import mysql.connector
import dotenv
import os
import datetime
import multiprocessing

import field_detection
import image_processing

def save_data(queue):
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
        video_informations = queue.get()
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
        start_time = queue.get()
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
            data = queue.get()
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
        end_time = queue.get()
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

def ball_tracking(queue):

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
        cv.imshow("frame", thresh)

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

if __name__ == "__main__":
    # create the queue to send data from one process to the other
    queue = multiprocessing.Queue()

    # create the procces to save the data into the database and start it
    db_process = multiprocessing.Process(target=save_data, args=(queue,))
    db_process.start()

    # create the ball tracking process and start it
    bt_process = multiprocessing.Process(target=ball_tracking, args=(queue,))
    bt_process.start()

    # wait for the processes to finish
    db_process.join()
    bt_process.join()