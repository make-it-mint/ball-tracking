# -- coding: utf-8 --

import numpy as np
import cv2 as cv
import math
from numba import jit

@jit(nopython=True)
def findCorner(image, x_start, y_start, vertical_orientation, horizontal_orientation, video_height, video_width):
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

def findLine(image, x, y, video_height, video_width):
    """
    Find the four end points of a line right of the starting point. 
    --------
    Keyword arguments:
    image -- binary image
    x -- x value of the start pixel
    y -- y value of the start pixel
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    valid_line -- boolean value to indicate if its a valid line (True if its Valid)
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

    valid_line = abs(abs(y[0] - y[2]) - abs(y[1] - y[3])) <= 2 and abs(abs(x[0] - x[1]) - abs(x[2] - x[3])) <= 2 and abs(y[0] - y[1]) > 100

    return valid_line, x, y

def lineCenter(x, y):
    """
    Calculate the upper and lower center points of the given line.
    --------
    Keyword arguments:
    x -- list wich contains the x values of the line end points
    y -- list wich contains the y values of the line end points
    --------
    Output:
    x -- list with the x values of the calculated points including the input points
    y -- list with the y values of the calculated points including the input points
    """

    # calculate the upper center point 5 between point 1 and 3
    x.append(round((x[0] + x[2]) / 2))
    y.append(round((y[0] + y[2]) / 2))

    # calculate the lower center point 6 between point 2 and 4
    x.append(round((x[1] + x[3]) / 2))
    y.append(round((y[1] + y[3]) / 2))

    return x, y

def checkUpperLine(image, x, y, video_height, video_width):
    """
    Find the lower end of the line. Then check if its the upper line.
    --------
    Keyword arguments:
    image -- binary image
    x -- list wich contains the x values of the line end points
    y -- list wich contains the y values of the line end points
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    upper_line -- boolean value to indicate if its the upper line (True if its the upper line)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """
    
    # set the start point to check if its the upper line
    x_check = x[5]
    y_check = y[5]

    # check if the start point is inside the frame
    if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
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

        # check if its the upper line and save the result in a variable
        upper_line = abs(abs(y[6] - y[5]) - abs(x[1] - x[3])) <= 3

        # delete the last point if its not the upper line
        if not upper_line:
            x.pop()
            y.pop()

        return upper_line, x, y
    else:
        return False, x, y

def checkLowerLine(image, x, y, video_height, video_width):
    """
    Find the upper end of the line. Then check if its the lower line.
    --------
    Keyword arguments:
    image -- binary image
    x -- list wich contains the x values of the line end points
    y -- list wich contains the y values of the line end points
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    lower_line -- boolean value to indicate if its the lower line (True if its the lower line)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """
    
    # set the start point to check if its the lower line
    x_check = x[4]
    y_check = y[4]

    # check if the start point is inside the frame
    if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
        # get the pixel value
        pixel_value = image[y_check, x_check]

        # loop to find the end of the lower line
        while pixel_value == 255 and y_check > 0 and y_check < video_height:
            # go one pixel up
            y_check -= 1
            # get the pixel value
            pixel_value = image[y_check, x_check]

        # save the found point 7
        x.append(x_check)
        y.append(y_check)

        # check if its the lower line and save the result in a variable
        lower_line = abs(abs(y[6] - y[4]) - abs(x[0] - x[2])) <= 3

        # delete the last point if its not the upper line
        if not lower_line:
            x.pop()
            y.pop()

        return lower_line, x, y
    else:
        return False, x, y

def checkLineCenter(image, x, y, video_height, video_width):
    """
    Find the center pointe between the upper and lower line. Then check if its the center of the field.
    --------
    Keyword arguments:
    image -- binary image
    x -- list wich contains the x values of the line end points
    y -- list wich contains the y values of the line end points
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    center -- boolean value to indicate if its the center of the field (True if its the center)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """

    # save the center point 15 of the field
    x.append(round((x[6] + x[13]) / 2))
    y.append(round((y[6] + y[13]) / 2))
    
    # calculate the endpoint for the radius of the circle
    xr = round((x[5] + x[6]) / 2)
    yr = round((y[5] + y[6]) / 2)

    # calculate the radius
    radius = math.sqrt(abs(xr - x[14])**2 + (abs(yr - y[14])**2))

    # create a list to save the pixel values on the circle
    values = []

    # loop to calculate the points on the circle and save the Pixel Values in the values list
    for theta in np.linspace(0, 2*math.pi, 100):
        # save the x and y values of the points
        #x.append(round(math.sin(theta)*radius+x[14]))
        #y.append(round(math.cos(theta)*radius+y[14]))
        x_radius = round(math.sin(theta)*radius+x[14])
        y_radius = round(math.cos(theta)*radius+y[14])
        # save the pixel values of the points
        values.append(image[y_radius, x_radius])

    # check if its the center of the field
    center = sum(values) / 255 > 50

    return center, x, y

def searchLine(image, x, y, line, video_height, video_width):
    """
    Find the second line. Then check if its the center of the field.
    --------
    Keyword arguments:
    image -- binary image
    x -- x value of the start pixel
    y -- y value of the start pixel
    line -- string to define wich line is searched ("lower" or "upper")
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    center_found -- boolean value to indicate if the center of the field is found (True if its the center)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """

    # set them variables if the lower line is searched
    if line == "lower":
        direction = 1
        x_start = x[5]
        y_start = y[5]

    # set them variables if the upper line is searched
    elif line == "upper":
        direction = -1
        x_start = x[4]
        y_start = y[4]

    # end the functioon if the input is incorrect
    else:
        return False, x, y
    
    # set the varable to indicate if the center is found to False
    center_found = False

    # calculate the angle of the line
    alpha = math.atan(abs(x[4]-x[5]) / abs(y[4]-y[5]))

    # set the point to search for the second line, depending on the angle and the length of the first line 
    x_check = round(x_start + math.sin(alpha) * abs(y[4]-y[5]) * 1.3 * direction)
    y_check = round(y_start + math.cos(alpha) * abs(y[4]-y[5]) * 1.3 * direction)

    # check if the points are inside the frame
    if x_check > 0 and x_check < video_width and y_check > 0 and y_check < video_height:
        # get the pixel value
        pixel_value = image[y_check, x_check]

        # Check if the pixel has a value of 255 (white)
        if pixel_value == 255:

            # go left while the pixel has value of 255 (white) and is inside the frame
            while pixel_value == 255 and x_check > 0 and x_check < video_width:

                # go one pixel to the left
                x_check -= 1
                
                # get the pixel value
                pixel_value = image[y_check, x_check]
            
            # start the search of the second line
            valid_line, x_second_line, y_second_line = findLine(image=image, x=x_check, y=y_check, video_height=video_height, video_width=video_width)

            # check if the line is valid
            if valid_line:
                # calculate the centerpoints of the line
                x_second_line, y_second_line = lineCenter(x_second_line, y_second_line)
                
                if line == "lower":
                    # check if its the lower line
                    line_check, x_second_line, y_second_line = checkLowerLine(image=image, x=x_second_line, y=y_second_line, video_height=video_height, video_width=video_width)
                elif line == "upper":
                    # check if its the upper line
                    line_check, x_second_line, y_second_line = checkUpperLine(image=image, x=x_second_line, y=y_second_line, video_height=video_height, video_width=video_width)

                # only continue if its the second line
                if line_check:

                    # save the points 
                    if line == "lower":
                        x.extend(x_second_line)
                        y.extend(y_second_line)
                    elif line == "upper":
                        x_second_line.extend(x)
                        y_second_line.extend(y)
                        x = x_second_line
                        y = y_second_line
                    
                    # check if its the center of the field
                    center, x, y = checkLineCenter(image=image, x=x, y=y, video_height=video_height, video_width=video_width)

                    # check if its the Circle in the center of the field
                    if center:
                        center_found = True

    return center_found, x, y

def getAngle(x, y):
    """
    Calculate the vertical angle and length of the center line.
    --------
    Keyword arguments:
    x -- list containing the x values of the center line points
    y -- list containing the y values of the center line points
    --------
    Output:
    angle -- angle of the center line in rad
    length -- length of the center line in pixels
    """

    # calculate the angle of the line
    alpha = math.atan(abs(x[4]-x[11]) / abs(y[4]-y[12]))
    
    # calculate the length of the line
    length = math.sqrt(abs(x[4]-x[11])**2 + abs(y[4]-y[12])**2)

    return alpha, length

def checkFieldCenter(image, x, y, video_height, video_width):
    """
    Calculate the upper and lower center points and find the lower end of the line. Then check if its the 
    upper line.
    --------
    Keyword arguments:
    image -- binary image
    x -- x value of the start pixel
    y -- y value of the start pixel
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    center_found -- boolean value to indicate if the center of the field is found (True if its the center)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """
    
    # set the variable to identify if the center is found to Falso for the case the center cant be found
    center_found = False

    # search the line and get the endpoints
    valid_line, x, y = findLine(image=image, x=x, y=y, video_height=video_height, video_width=video_width)

    # check if the line is valid
    if valid_line:
        # calculate the centerpoints of the line
        x, y = lineCenter(x, y)

        # check if its the upper line
        upper_line, x, y = checkUpperLine(image=image, x=x, y=y, video_height=video_height, video_width=video_width)
        
        # search the lower line if its the upper line
        if upper_line:
            # search for the lower line and check if ist the center of the field
            center_found, x, y = searchLine(image=image, x=x, y=y, line="lower", video_height=video_height, video_width=video_width)
        
        else:
            # check if its the lower line
            lower_line, x, y = checkLowerLine(image=image, x=x, y=y, video_height=video_height, video_width=video_width)

            # search the upper line if its the lower line
            if lower_line:
                # search for the upper line and check if ist the center of the field
                center_found, x, y = searchLine(image=image, x=x, y=y, line="upper", video_height=video_height, video_width=video_width)
        
    return center_found, x, y

def findPenaltyRoom(image, x, y, angle, length, video_height, video_width):
    """
    Search the penalty room from the center of the field.
    --------
    Keyword arguments:
    image -- binary image
    x -- list containing the x values of the center line points
    y -- list containing the y values of the center line points
    angle -- angle of the center line in rad
    length -- length of the center line in pixels
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    center_found -- boolean value to indicate if the center of the field is found (True if its the center)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """

    # calculate the starting point to search the corner
    x_check = round(x[14] - length * 0.51)
    y_check = round(y[14] - length * 0.26)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="up", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] - length * 0.51)
    y_check = round(y[14] + length * 0.26)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="down", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] - length * 0.72)
    y_check = round(y[14] - length * 0.16)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="up", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] - length * 0.72)
    y_check = round(y[14] + length * 0.16)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="down", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] + length * 0.51)
    y_check = round(y[14] - length * 0.26)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="up", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] + length * 0.51)
    y_check = round(y[14] + length * 0.26)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="down", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] + length * 0.72)
    y_check = round(y[14] - length * 0.16)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="up", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    # calculate the starting point to search the corner
    x_check = round(x[14] + length * 0.72)
    y_check = round(y[14] + length * 0.16)
    # search the corner
    x_corner, y_corner = findCorner(image=image, x_start=x_check, y_start=y_check, vertical_orientation="down", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
    # save the corner coordinates
    x.append(x_corner)
    y.append(y_corner)

    return x, y

def findField(image, video_height, video_width):
    """
    Search in the Frame for the field. The frame is scanned in horizontal lines until a white pixel is reached, 
    then at this the center line and the center of the field is searched. After the center line of the field is found,
    the penalty room is searched from the center of the field.
    --------
    Keyword arguments:
    image -- binary image
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    center_found -- boolean value to indicate if the center of the field is found (True if its the center)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """

    x_border_left = round(video_width * 0.30)
    x_border_right = video_width - x_border_left

    y_border_upper = round(video_height * 0.20)
    y_border_lower = video_height - y_border_upper
    


    # define the start values for x and y
    x_check = x_border_left
    y_check = y_border_upper

    # set the variable to indicate if the center is found to False
    center_found = False
    
    # while loop to find the center of the field
    while not center_found:
        # go to the next pixel to the right
        x_check += 1

        # get the pixel value
        pixel_value = image[y_check, x_check]

        # check if the ende of the frame is reached 
        if x_check >= (x_border_right) and y_check > (y_border_lower):
            return center_found, [], []
        
        # check if the right border of the frame is reached
        elif x_check >= (x_border_right):
            # reset the x value and go 100 pixels down
            x_check = x_border_left
            y_check += 200
        
        # check if the pixel has a value of 255 (white)
        if pixel_value == 255:

            # try to find the center 
            center_found, x, y = checkFieldCenter(image=image, x=x_check, y=y_check, video_height=video_height, video_width=video_width)

            # check if the center is found
            if center_found:
                # calculate the angle of the center line
                angle, length = getAngle(x=x, y=y)

                # find the corner points of the penalty room
                x, y = findPenaltyRoom(image=image, x=x, y=y, angle=angle, length=length, video_height=video_height, video_width=video_width)

                return center_found, x, y
            
            # go to the other side if the white object if the center is not found
            else:
                # go to the other side until a pixel with value 0 (black) is reached
                while pixel_value == 255 and x_check < (x_border_right):
                    # go one pixel to the right
                    x_check += 1

                    # get the pixel value
                    pixel_value = image[y_check, x_check]

def fielDetection(image, x_old, y_old, field_found, video_height, video_width):
    """
    Depending if the field position is known in the previous frame, check if th positions are still the same or 
    search for the field. If somme positions are not at the same position, they are substituted with the positions 
    from the previous frame.
    --------
    Keyword arguments:
    image -- binary image
    x_old -- x values of the field points from the previous frame
    y_old -- y values of the field points from the previous frame
    field_found -- boolean value to indicate if the  field is found (True if the field is found)
    video_height -- height of the frame in pixels
    video_width -- width of the frame in pixels
    --------
    Output:
    center_found -- boolean value to identify if the center of the field is found (True if its the center)
    x -- list with the x values of the detected points including the input points
    y -- list with the y values of the detected points including the input points
    """
    field_moved = False

    # check if the field is found in the previous frame
    if field_found:

        check_distance = 5
        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[0]-check_distance), y_start=(y_old[0]+check_distance), vertical_orientation="up", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 1
        x = [x_corner]
        y = [y_corner]

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[1]-check_distance), y_start=(y_old[1]-check_distance), vertical_orientation="down", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 2
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[2]+check_distance), y_start=(y_old[2]+check_distance), vertical_orientation="up", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 3
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[2]+check_distance), y_start=(y_old[2]-check_distance), vertical_orientation="down", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 4
        x.append(x_corner)
        y.append(y_corner)

        # find the center points 5 and 6 of the line
        x, y = lineCenter(x, y)

        # starting point to search for point 7
        x_check = x[5]
        y_check = y[5]

        # get the pixel value
        pixel_value = image[y_check, x_check]

        # go to the end of the line
        while pixel_value == 255 and y_check > 0 and y_check < video_height:
            # go one pixel down
            y_check += 1
            # get the pixel value
            pixel_value = image[y_check, x_check]
        
        # save point 7
        x.append(x_check)
        y.append(y_check)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[7]-check_distance), y_start=(y_old[7]+check_distance), vertical_orientation="up", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 8
        x2 = [x_corner]
        y2 = [y_corner]

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[8]-check_distance), y_start=(y_old[8]-check_distance), vertical_orientation="down", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 9
        x2.append(x_corner)
        y2.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[9]+check_distance), y_start=(y_old[9]+check_distance), vertical_orientation="up", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 10
        x2.append(x_corner)
        y2.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[10]+check_distance), y_start=(y_old[10]-check_distance), vertical_orientation="down", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 11
        x2.append(x_corner)
        y2.append(y_corner)

        # find the center points 12 and 13 of the line
        x2, y2 = lineCenter(x2, y2)

        # starting point to search for point 14
        x_check = x2[5]
        y_check = y2[5]

        # get the pixel value
        pixel_value = image[y_check, x_check]

        while pixel_value == 255 and y_check > 0 and y_check < video_height:
            y_check += 1
            # get the pixel value
            pixel_value = image[y_check, x_check]
        
        # save point 14
        x2.append(x_check)
        y2.append(y_check)

        # save the points of the lower line in the variable with the points of the upper line
        x.extend(x2)
        y.extend(y2)

        # calculate and save the center point 15
        x.append(round((x[6] + x[13]) / 2))
        y.append(round((y[6] + y[13]) / 2))

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[15]-check_distance), y_start=(y_old[15]+check_distance), vertical_orientation="up", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 16
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[16]-check_distance), y_start=(y_old[16]-check_distance), vertical_orientation="down", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 17
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[17]-check_distance), y_start=(y_old[17]+check_distance), vertical_orientation="up", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 18
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[18]-check_distance), y_start=(y_old[18]-check_distance), vertical_orientation="down", horizontal_orientation="right", 
                                    video_height=video_height, video_width=video_width)
        # save point 19
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[19]+check_distance), y_start=(y_old[19]+check_distance), vertical_orientation="up", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 20
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[20]+check_distance), y_start=(y_old[20]-check_distance), vertical_orientation="down", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 21
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[21]+check_distance), y_start=(y_old[21]+check_distance), vertical_orientation="up", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 22
        x.append(x_corner)
        y.append(y_corner)

        # search for the corner on the old position
        x_corner, y_corner = findCorner(image=image, x_start=(x_old[22]+check_distance), y_start=(y_old[22]-check_distance), vertical_orientation="down", horizontal_orientation="left", 
                                    video_height=video_height, video_width=video_width)
        # save point 23
        x.append(x_corner)
        y.append(y_corner)

        # set the variables to check how manny points are the same and wich points are different
        points_found = 0
        missed_points = []

        # check how many points are at the same position as in the previous frame
        for i in range(len(x)):
            # compare the x and y positions
            if abs(x[i] - x_old[i]) < 2 and abs(y[i] - y_old[i]) < 2:
                # indicate that the point is at the same position
                points_found += 1

            else:
                # save the indx if the positions dont match
                missed_points.append(i)

        # check if all the points are at the same position as in the previous frame
        if points_found == len(x):
            # the field is found if all the positions are the same
            field_found = True

        # check if a certain amount of points is at the same position
        elif points_found > (len(x) - 15):
            # substitude the positions that dont match with the positions of the previous frame
            for i in missed_points:
                x[i] = x_old[i]
                y[i] = y_old[i]

            # calculate and substitude the center point 15
            x[14] = round((x[6] + x[13]) / 2)
            y[14] = round((y[6] + y[13]) / 2)

            # the field is found
            field_found = True
        
        # if the field cant be found try to find the field with the findField function
        else:
            # try to find the field
            field_found, x, y = findField(image=image, video_height=video_height, video_width=video_width)
            field_moved = True
    
    # if the field is unknown in the previous frame try to find it 
    else:
        # try to find the field
        field_found, x, y = findField(image=image, video_height=video_height, video_width=video_width)
        field_moved = True


    return field_found,field_moved, x, y



        
            









