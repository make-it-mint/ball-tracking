# -- coding: utf-8 --

import numpy as np
import cv2 as cv

def findTreshold(image):

    # convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # calculate the grayscale histogram
    hist = cv.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=(0, 256), accumulate=False)

    # normalize the histogram
    hist_normalized = (hist / sum(hist))

    # set starting values for the variables
    percentage = 1
    treshold = 255

    # numerical integration until the percentage is reached
    while percentage >= 0.93:
        treshold -= 1
        percentage = 1 - sum(hist_normalized[treshold:-1])

    return treshold

    