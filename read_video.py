# -- coding: utf-8 --

import numpy as np
import cv2 as cv

# video capturing from video file or camera
# to read a video file insert the file name
# for a camera insert an integer depending on the camera port
cap = cv.VideoCapture("videos/test-game.mp4")

fps = cap.get(cv.CAP_PROP_FPS)
print(f"fps: {fps}")
frame_time = int(1000/fps)

# exit the programm if the camera cannot be oppend, or the video file cannot be read
if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    # stop the loop when the frame is not read correctly
    if not  ret:
        print("Can't recive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    """Harris Corner Detection"""
    """
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    frame[dst>0.01*dst.max()]=[0,0,255]
    """

    """Corner with SupPixel Accuracy"""
    """
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    res0 = res[:,0]
    res0[res0 >= 1920] = 1919
    res1 = res[:,1]
    res1[res1 >= 1080] = 1079
    res2 = res[:,2]
    res2[res2 >= 1920] = 1919
    res3 = res[:,3]
    res3[res3 >= 1080] = 1079
    
    frame[res1,res0]=[0,0,255]
    frame[res3,res2] = [0,255,0]
    """

    """Shi-Tomasi Corner Detector"""
    """
    # (good for tracking) 
    # find 100 best corners
    corners = cv.goodFeaturesToTrack(gray,100,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv.circle(frame,(x,y),3,255,-1)
    """
    
    """SIFT Scale-Invariant Feature Transform"""
    """
    sift = cv.SIFT_create()

    #kp = sift.detect(gray,None)
    #kp,des = sift.compute(gray,kp)

    # find keypoints and descriptors in one step (same as the two lines above)
    kp, des = sift.detectAndCompute(gray,None)

    frame=cv.drawKeypoints(gray,kp,frame,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    """

    """SURF Speeded-Up Robust Features"""
    """
    # does not work algorithm is patented and therefore excluded
    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    surf = cv.xfeatures2d.SURF_create(400)

    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(frame,None)

    frame = cv.drawKeypoints(frame,kp,None,(255,0,0),4)
    """

    """FAST"""
    """
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # Disable nonmaxSuppression
    #fast.setNonmaxSuppression(0)

    # find and draw the keypoints
    kp = fast.detect(frame,None)
    frame = cv.drawKeypoints(frame, kp, None, color=(255,0,0))

    # Print all default params
    #print( "Threshold: {}".format(fast.getThreshold()) )
    #print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    #print( "neighborhood: {}".format(fast.getType()) )
    #print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    """

    """BRIEF Binary Robust Independent Elementary Features"""
    """
    # opencv contributor pack is needed for this
    # Initiate STAR detector
    star = cv.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(frame,None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(frame, kp)

    # draw the keypoints
    frame = cv.drawKeypoints(frame, kp, None, color=(255,0,0))
    """

    """ORB Orient FAST and Rotated BRIEF"""

    # Initiate ORB detector
    orb = cv.ORB_create()

    orb.setMaxFeatures(500)

    # find the keypoints with ORB
    #kp = orb.detect(frame,None)

    # compute the descriptors with ORB
    #kp, des = orb.compute(frame, kp)

    # Find keypoints and descriptors directly
    kp, des = orb.detectAndCompute(frame,None)

    # draw only keypoints location,not size and orientation
    frame = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    # Display the resulting frame
    cv.imshow("frame", frame)

    # stop the loop if the "q" key on the keyboard is pressed 
    if cv.waitKey(frame_time) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
