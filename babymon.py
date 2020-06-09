# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from imutils.io import TempFile
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
from config import Config
from twilionotifier import TwilioNotifier
from datetime import datetime, timedelta
from datetime import date


def send_alert():
    pass


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
                help="path to the input configurtion file")
args = vars(ap.parse_args())
config = Config(args["config"])
tn = TwilioNotifier(config)

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = config["eye_aspect_ratio_thresh"]
OPEN_THRESH = config["open_thresh_seconds"]
CLOSED_THRESH = config["closed_thresh_seconds"]
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
EYES_OPEN_COUNTER = 0
EYES_CLOSED_COUNTER = 0
ALERT_ON = False

SKIP_FRAME = 5
COUNT_FRAMES = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=config["webcam"]).start()
time.sleep(1.0)
# loop over frames from the video stream

startTime = None
endTime = None
lastSentTime = datetime.now() - timedelta(minutes=5)

while True:
    # read frame and convert image to grayscale
    COUNT_FRAMES += 1
    if COUNT_FRAMES % SKIP_FRAME != 0:
        continue
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face using dlib
    rects = detector(gray, 0)

    # draw eyes for every face detected
    for rect in rects:
        # predict 68 point landmarks using dlib
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart: lEnd]
        rightEye = shape[rStart:rEnd]
        
        # calculate average eye aspect ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = np.average([leftEAR, rightEAR])

        # find convex hull of left and right eyes and draw contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # eyes open        
        if ear >= EYE_AR_THRESH:
            if not startTime:
                startTime = datetime.now()
            EYES_CLOSED_COUNTER = 0
            EYES_OPEN_COUNTER += 1
            # if the eyes were closed for a sufficient number of
			# then sound the alarm
            if EYES_OPEN_COUNTER >= OPEN_THRESH:
                timeDiff = (datetime.now() - startTime).seconds
                if not ALERT_ON:
                    ALERT_ON = True
                    # record temp video
                    tempVideo = TempFile(ext=".mp4")
                    print(tempVideo.path)
                    writer = cv2.VideoWriter(tempVideo.path, 0x21, 30, (300,300), True)                                

                elif ALERT_ON and timeDiff >= 10 and lastSentTime <= datetime.now() - timedelta(minutes=5):
                    writer.release()
                    writer = None
                    msg = "Baby is awake"
                    print("sending message to parent")
                    tn.send(msg, tempVideo)
                    lastSentTime = datetime.now()
                cv2.putText(frame, "BABY IS AWAKE!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # eyes closed
        else:
            EYES_CLOSED_COUNTER += 1
            if EYES_CLOSED_COUNTER >= CLOSED_THRESH:
                EYES_OPEN_COUNTER = 0
                ALERT_ON = False
                cv2.putText(frame, "Zzzzzzz!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(frame, "BABY IS AWAKE!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # display eye aspect ratio
        cv2.putText(frame, "EAR = {:.2f}".format(ear), (300,30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
