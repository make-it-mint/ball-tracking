# -- coding: utf-8 --

import numpy as np
import cv2 as cv

def findTreshold(image):

    # convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # calculate the grayscale histogram
    hist = cv.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=(0, 256), accumulate=False)

    return hist

    