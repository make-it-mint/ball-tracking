# -- coding: utf-8 --

import numpy as np
import cv2 as cv

def findCorner(image, x_start, y_start, vertical_orientation, horizontal_orientation, video_height, video_width):
    """
    Find a corner from a starting point in the choosen directions.

    Keyword arguments:
    image -- binary image
    x_start -- x value of the start pixel
    y_start -- y value of the start pixel
    vertical_orientation -- vertical direction to search the corner ("up" or "down")
    horizontal_orientation -- horizontal direction to search the corner ("left" or "right")
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels

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
            y_check = y + (3 * step_y)

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

                # reset the y_check value
                y_check = y + (3 * step_y)

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

def findLine(image, x, y, video_height, video_width):
    """
    Find the four end points of a line right of the starting point. 

    Keyword arguments:
    image -- binary image
    x -- x value of the start pixel
    y -- y value of the start pixel
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels

    Output:
    x -- list with the x values of the four points
    y -- list with the y values of the four points
    """

    # search the upper left point
    x1, y1 = findCorner(image=image, x_start=x, y_start=y, vertical_orientation="up", horizontal_orientation="right", video_height=video_height, video_width=video_width)
    # search the lower left point
    x2, y2 = findCorner(image=image, x_start=x, y_start=y, vertical_orientation="down", horizontal_orientation="right", video_height=video_height, video_width=video_width)

    # get the pixel value
    pixel_value = image[y, x]

    # go to the line
    while pixel_value == 0:
        # go to the next pixel
        x += 1 
        # get the pixel value
        pixel_value = image[y, x]

        # stop the loop if the border of the frame is reached
        if x == video_width:
            # go one pixel back
            x -= 1
            break
    
    # go to the oder side of the line
    while pixel_value == 255:
        # go to the next pixel
        x += 1 
        # get the pixel value
        pixel_value = image[y, x]

        # stop the loop if the border of the frame is reached
        if x == video_width:
            # go one pixel back
            x -= 1
            break

    # search the upper right corner
    x3, y3 = findCorner(image=image, x_start=x, y_start=y, vertical_orientation="up", horizontal_orientation="left", video_height=video_height, video_width=video_width)
    # search the lower right corner
    x4, y4 = findCorner(image=image, x_start=x, y_start=y, vertical_orientation="down", horizontal_orientation="left", video_height=video_height, video_width=video_width)

    # define the lists with the x and y values
    x = [x1, x2, x3, x4]
    y = [y1, y2, y3, y4]

    return x, y

def checkUpperLine(image, x, y, video_height, video_width):
    """
    Calculate the upper and lower center points and find the lower end of the line. Then check if its the upper line.

    Keyword arguments:
    image -- binary image
    x -- list with the x values of the endpoints of the line
    y -- list with the y values of the endpoints of the line
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels

    Output:
    upper_line -- boolean value to identify if its the upper line (True if its the upper line)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """
    # calculate the upper center point 5 between point 1 and 3
    x.append(round((x[0] + x[2]) / 2))
    y.append(round((y[0] + y[2]) / 2))

    # canlculate the lower center point 6 between point 2 and 4
    x.append(round((x[1] + x[3]) / 2))
    y.append(round((y[1] + y[3]) / 2))

    # set the start point to check if its the upper line
    x_check = x[5]
    y_check = y[5]

    # get the pixel value
    pixel_value = image[y_check, x_check]

    # loop to find the end of the upper line
    while pixel_value == 255 and y_check > 0 and y_check < video_height:
        # go one pixel down
        y_check += 1
        # get the pixel value
        pixel_value = image[y_check, x_check]

    # save the found point 7
    x.append(x_check)
    y.append(y_check)

    # check if its the upper line
    upper_line = abs(abs(y[6] - y[5]) - abs(x[1] - x[3])) <= 2

    return upper_line, x, y


