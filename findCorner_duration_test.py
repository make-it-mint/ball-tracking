# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import datetime
import matplotlib.pyplot as plt
from numba import jit

def findCornerNormal(image, x_start, y_start, vertical_orientation, horizontal_orientation, video_height, video_width):

    """
    Find a corner from a starting point in the choosen directions.
    --------
    Keyword arguments:
    image -- binary image
    x_start -- x value of the start pixel
    y_start -- y value of the start pixel
    vertical_orientation -- vertical direction to search the corner ("up" or "down")
    horizontal_orientation -- horizontal direction to search the corner ("left" or "right")
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    x -- x value of the corner in pixels
    y -- y value of the corner in pixels
    """

    # define the vertical direction
    if vertical_orientation == "up":
        step_y = -1
    else:
        step_y = 1

    # define the horizontal direction
    if horizontal_orientation == "right":
        step_x = 1
    else:
        step_x = -1
    
    # save the start values in variables
    x = x_start
    y = y_start

    # define variable for the while loop
    corner_reached = False
    i = 0

    # while loop to find the corner in vertical direction
    while not corner_reached and x >= 0 and x <= video_width and y >= 0 and y <= video_height:
        # get the pixel value for the start pixel
        pixel_value = image[y, x]

        # only while the pixel is inside the frame go left or right depending on the choosen direction until a pixel with value 1 (white) is reached  
        while pixel_value == 0 and x > 0 and x < video_width:
            # go to the next pixel in the choosen direction
            x += step_x
            # get the pixel value
            pixel_value = image[y, x]
        
        # go one pixel back in horizontal direction to the pixel with value 0 (black)
        x -= step_x

        # check if x is still inside the frame
        if x < 0 or x > video_width:
            break

        # get the pixel value
        pixel_value = image[y, x]

        # only while the pixel is inside the frame go up or down depending on the choosen direction until a pixel, with value 1 (white) is reached 
        while pixel_value == 0 and y > 0 and y < video_height:
            # go to the next pixel in the choosen direction
            y += step_y
            # get the pixel value
            pixel_value = image[y, x]
        
        # break the while-loop if the pixel is on the left or right border of the frame
        if x <= 0 or x >= video_width:
            break

        # else check if the next pixel in horizontal direction is 0 (black)
        elif image[y, (x - step_x)] == 0:
            # go to the next pixel in teh opposite horizontal direction
            x -= step_x

        # else a corner is reached, go one pixel back in vertical direction
        else:
            # go one pixel back in vertical direction to the pixel with value 0 (black)
            y -= step_y

            # check if the corner is the ball
            # define the starting points to search for the first width
            x_check = x
            y_check = y - (5 * step_y)

            # check if the values are inside the frame 
            if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
                # get the pixel value
                pixel_value = image[y_check, x_check]

                # search in horizontal direction for the next pixel with value 1 (white) to find the line
                while pixel_value == 0 and x_check > 0 and x_check < video_width:
                    # go to the next pixen in horizontal direction
                    x_check += step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
                
                # save the start of the line in a variable
                x1 = x_check - step_x
                
                # search in horizontal direction for the next pixel with value 0 (black) measure the line thickness
                while pixel_value == 255 and x_check > 0 and x_check < video_width:
                    # go to the next pixen in horizontal direction
                    x_check += step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
            
                # calculate the width of the line and save it in a variable
                width1 = abs(x1 - x_check)
            
            # set the width1 variable to 0 if the pixel is outside the frame
            else:
                width1 = 0

            # define the starting points to search for the second line
            x_check = x
            y_check = y + (5 * step_y)

            # check if the values are inside the frame 
            if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
                # get the pixel value
                pixel_value = image[y_check, x_check]

                # search in opposite horizontal direction for the next pixel with value 0 (black) to find the start position of the second width
                while pixel_value == 255 and x_check > 0 and x_check < video_width:
                    # go to the next pixel in opposite horizontal direction
                    x_check -= step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
            
                # save the start of the potential ball in a variable
                x1 = x_check

                # reset the x_check value
                x_check = x + (3 * step_x)

                # get the pixel value
                pixel_value = image[y_check, x_check]

                # search in horizontal direction for the next pixel with value 0 (black) to measure the second width
                while pixel_value == 255 and x_check > 0 and x_check < video_width:
                    # go to the next pixel in opposite horizontal direction
                    x_check += step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
                
                # calculate the width of the potential ball and save it in a variable
                width2 = abs(x1 - x_check)
            
            # set the width2 variable to 0 if the pixel is outside the frame
            else:
                width2 = 0
            
            # set the staring positions to find the rest length
            x_check = round(x + (width1 * step_x / 2))
            y_check = y

            # check if the values are inside the frame 
            if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
                # get the pixel value
                pixel_value = image[y_check, x_check]

                while pixel_value == 255 and y_check > 0 and y_check < video_height:
                    # go to the next pixel in vertical direction
                    y_check += step_y
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
                
                rest_length = abs(y - y_check)
            
            # set the rest_length variable to 0 if the pixel is outside the frame
            else:
                rest_length = 0
            
            # check if the pixels are far enough away from the borders of the frame for the following checks
            if x > 2 and x < (video_width - 2) and y > 2 and y < (video_height - 2):

                # check if the object on the line is the ball
                if width2 > (width1 * 1.5) and width2 < (width1 * 6) and rest_length > (width1 * 2):
                    # set the start values to find the other side of the ball
                    x -= 3 * step_x
                    y += 3 * step_y
                    
                    # get the pixel value
                    pixel_value = image[y, x]

                    # loop to go to the other side of the ball
                    while pixel_value == 255 and y > 0 and y <video_height:
                        # go to the next pixel in vertical direction
                        y += step_y
                        # get the pixel value
                        pixel_value = image[y, x]
                
                # check if if the corner is not reached
                elif image[(y - step_y), (x + (2 * step_x))] == 0:
                    # set new x and y values for the next iteration of the loop
                    x += 2 * step_x
                    y -= step_y 
                
                # check if the corner is reached from the first condition
                elif image[y, (x + step_x)] == 1 and image[(y - step_y), (x + step_x)] == 0:
                    # change x to the center of the corner
                    x += step_x
                    # set the variable for the loop to True to stop the while-loop
                    corner_reached = True

                # check if the corner is reached from the second condition
                elif image[y, (x + step_x)] == 1:
                    # set the variable for the loop to True to stop the while-loop
                    corner_reached = True
        
        # break the loop after 10 iterations
        if i == 10:
            break
        
        # count the iterations
        i += 1

    # let the function return the x and y value of the corner
    return x, y

@jit(nopython=True)
def findCornerCompiled(image, x_start, y_start, vertical_orientation, horizontal_orientation, video_height, video_width):
    """
    Find a corner from a starting point in the choosen directions.
    --------
    Keyword arguments:
    image -- binary image
    x_start -- x value of the start pixel
    y_start -- y value of the start pixel
    vertical_orientation -- vertical direction to search the corner ("up" or "down")
    horizontal_orientation -- horizontal direction to search the corner ("left" or "right")
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    x -- x value of the corner in pixels
    y -- y value of the corner in pixels
    """

    # define the vertical direction
    if vertical_orientation == "up":
        step_y = -1
    else:
        step_y = 1

    # define the horizontal direction
    if horizontal_orientation == "right":
        step_x = 1
    else:
        step_x = -1
    
    # save the start values in variables
    x = x_start
    y = y_start

    # define variable for the while loop
    corner_reached = False
    i = 0

    # while loop to find the corner in vertical direction
    while not corner_reached and x >= 0 and x <= video_width and y >= 0 and y <= video_height:
        # get the pixel value for the start pixel
        pixel_value = image[y, x]

        # only while the pixel is inside the frame go left or right depending on the choosen direction until a pixel with value 1 (white) is reached  
        while pixel_value == 0 and x > 0 and x < video_width:
            # go to the next pixel in the choosen direction
            x += step_x
            # get the pixel value
            pixel_value = image[y, x]
        
        # go one pixel back in horizontal direction to the pixel with value 0 (black)
        x -= step_x

        # check if x is still inside the frame
        if x < 0 or x > video_width:
            break

        # get the pixel value
        pixel_value = image[y, x]

        # only while the pixel is inside the frame go up or down depending on the choosen direction until a pixel, with value 1 (white) is reached 
        while pixel_value == 0 and y > 0 and y < video_height:
            # go to the next pixel in the choosen direction
            y += step_y
            # get the pixel value
            pixel_value = image[y, x]
        
        # break the while-loop if the pixel is on the left or right border of the frame
        if x <= 0 or x >= video_width:
            break

        # else check if the next pixel in horizontal direction is 0 (black)
        elif image[y, (x - step_x)] == 0:
            # go to the next pixel in teh opposite horizontal direction
            x -= step_x

        # else a corner is reached, go one pixel back in vertical direction
        else:
            # go one pixel back in vertical direction to the pixel with value 0 (black)
            y -= step_y

            # check if the corner is the ball
            # define the starting points to search for the first width
            x_check = x
            y_check = y - (5 * step_y)

            # check if the values are inside the frame 
            if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
                # get the pixel value
                pixel_value = image[y_check, x_check]

                # search in horizontal direction for the next pixel with value 1 (white) to find the line
                while pixel_value == 0 and x_check > 0 and x_check < video_width:
                    # go to the next pixen in horizontal direction
                    x_check += step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
                
                # save the start of the line in a variable
                x1 = x_check - step_x
                
                # search in horizontal direction for the next pixel with value 0 (black) measure the line thickness
                while pixel_value == 255 and x_check > 0 and x_check < video_width:
                    # go to the next pixen in horizontal direction
                    x_check += step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
            
                # calculate the width of the line and save it in a variable
                width1 = abs(x1 - x_check)
            
            # set the width1 variable to 0 if the pixel is outside the frame
            else:
                width1 = 0

            # define the starting points to search for the second line
            x_check = x
            y_check = y + (5 * step_y)

            # check if the values are inside the frame 
            if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
                # get the pixel value
                pixel_value = image[y_check, x_check]

                # search in opposite horizontal direction for the next pixel with value 0 (black) to find the start position of the second width
                while pixel_value == 255 and x_check > 0 and x_check < video_width:
                    # go to the next pixel in opposite horizontal direction
                    x_check -= step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
            
                # save the start of the potential ball in a variable
                x1 = x_check

                # reset the x_check value
                x_check = x + (3 * step_x)

                # get the pixel value
                pixel_value = image[y_check, x_check]

                # search in horizontal direction for the next pixel with value 0 (black) to measure the second width
                while pixel_value == 255 and x_check > 0 and x_check < video_width:
                    # go to the next pixel in opposite horizontal direction
                    x_check += step_x
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
                
                # calculate the width of the potential ball and save it in a variable
                width2 = abs(x1 - x_check)
            
            # set the width2 variable to 0 if the pixel is outside the frame
            else:
                width2 = 0
            
            # set the staring positions to find the rest length
            x_check = round(x + (width1 * step_x / 2))
            y_check = y

            # check if the values are inside the frame 
            if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
                # get the pixel value
                pixel_value = image[y_check, x_check]

                while pixel_value == 255 and y_check > 0 and y_check < video_height:
                    # go to the next pixel in vertical direction
                    y_check += step_y
                    # get the pixel value
                    pixel_value = image[y_check, x_check]
                
                rest_length = abs(y - y_check)
            
            # set the rest_length variable to 0 if the pixel is outside the frame
            else:
                rest_length = 0
            
            # check if the pixels are far enough away from the borders of the frame for the following checks
            if x > 2 and x < (video_width - 2) and y > 2 and y < (video_height - 2):

                # check if the object on the line is the ball
                if width2 > (width1 * 1.5) and width2 < (width1 * 6) and rest_length > (width1 * 2):
                    # set the start values to find the other side of the ball
                    x -= 3 * step_x
                    y += 3 * step_y
                    
                    # get the pixel value
                    pixel_value = image[y, x]

                    # loop to go to the other side of the ball
                    while pixel_value == 255 and y > 0 and y <video_height:
                        # go to the next pixel in vertical direction
                        y += step_y
                        # get the pixel value
                        pixel_value = image[y, x]
                
                # check if if the corner is not reached
                elif image[(y - step_y), (x + (2 * step_x))] == 0:
                    # set new x and y values for the next iteration of the loop
                    x += 2 * step_x
                    y -= step_y 
                
                # check if the corner is reached from the first condition
                elif image[y, (x + step_x)] == 1 and image[(y - step_y), (x + step_x)] == 0:
                    # change x to the center of the corner
                    x += step_x
                    # set the variable for the loop to True to stop the while-loop
                    corner_reached = True

                # check if the corner is reached from the second condition
                elif image[y, (x + step_x)] == 1:
                    # set the variable for the loop to True to stop the while-loop
                    corner_reached = True
        
        # break the loop after 10 iterations
        if i == 10:
            break
        
        # count the iterations
        i += 1

    # let the function return the x and y value of the corner
    return x, y

video_file = "ball_tracking_test.mp4"

# video capturing from video file or camera
# to read a video file insert the file name
# for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("Test-Videos/" + video_file)

# exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

video_width = int(cap.get(3))
video_height = int(cap.get(4))

# reduce the video width and heigth to match the max index
video_height -= 1
video_width -= 1

times_normal = []
times_compilated = []
frames = []
frame_count = 0

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    # stop the loop when the frame is not read correctly
    if not ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    _, thresh = cv.threshold(gray, 175, 255, cv.THRESH_BINARY)

    start_time_normal = datetime.datetime.now()

    x = []
    y = []
    for _ in range(100):
        x_corner, y_corner = findCornerNormal(image=thresh, x_start=900, y_start=800, vertical_orientation="up", horizontal_orientation="right", video_height=video_height, video_width=video_width)
        x.append(x_corner)
        y.append(y_corner)

    end_time_normal = datetime.datetime.now()

    start_time_compilated = datetime.datetime.now()

    x = []
    y = []
    for _ in range(100):
        x_corner, y_corner = findCornerCompiled(image=thresh, x_start=900, y_start=800, vertical_orientation="up", horizontal_orientation="right", video_height=video_height, video_width=video_width)
        x.append(x_corner)
        y.append(y_corner)

    end_time_compilated = datetime.datetime.now()

    duration_normal = end_time_normal - start_time_normal
    duration_normal = duration_normal.total_seconds() * 100

    duration_compilated = end_time_compilated - start_time_compilated
    duration_compilated = duration_compilated.total_seconds() * 100

    times_normal.append(duration_normal)
    times_compilated.append(duration_compilated)
    frames.append(frame_count)
    
    frame_count += 1
    
    cv.imshow("thresh", frame)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

print(f"Anzahl der Frames:\n{frames[-1]+1}\n")

print(f"Dauer des ersten Frames ohne Vorkompilierung: \n{times_normal[0]}ms")
print(f"durchschnittliche Dauer ohne Vorkompilierung: \n{np.mean(times_normal)}ms")
print(f"durchschnittliche Dauer ohne den ersten Frame ohne Vorkompilierung: \n{np.mean(times_normal[1:-1])}ms\n")

print(f"Dauer des ersten Frames mit Vorkompilierung: \n{times_compilated[0]}ms")
print(f"durchschnittliche Dauer mit Vorkompilierung: \n{np.mean(times_compilated)}ms")
print(f"durchschnittliche Dauer ohne den ersten Frame mit Vorkompilierung: \n{np.mean(times_compilated[1:-1])}ms\n")

print(f"Verringerung der Dauer durch Vorkompilierung:\n{(1-np.mean(times_compilated[1:-1])/np.mean(times_normal[1:-1]))*100}")

plt.plot(frames, times_normal, "-b", label="ohne Vorkompilierung")
plt.plot(frames, times_compilated, "-r", label="mit Vorkompilierung")
plt.legend(loc="upper right")
plt.title("Dauer von 100x findCorner mit und ohne Vorkompilierung")
plt.ylabel("Zeit [ms]")
plt.xlabel("Frame")
plt.xlim([-1, frames[-1]])
plt.grid()
plt.show()

