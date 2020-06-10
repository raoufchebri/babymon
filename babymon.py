# USAGE
# python babymon.py --config config/config.json

# import the necessary packages
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from imutils.io import TempFile
import imutils
import argparse
import time
from utils.config import Config
from notifications.twilionotifier import TwilioNotifier
from datetime import datetime, timedelta, date
import sys
import signal

# function to handle keyboard interrupt
def signal_handler(sig, frame):
	print("[INFO] You pressed `ctrl + c`! Closing mail detector" \
		" application...")
	sys.exit(0)

# function to calculate the eye aspect ratio
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

def send_alert(msg: str, img):
    temp_file = TempFile(ext=".jpg")
    cv2.imwrite(temp_file.path, img)
    tn.send(msg, temp_file)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
                help="path to the input configurtion file")
args = vars(ap.parse_args())
# construct the config object and twilio notifier
config = Config(args["config"])
tn = TwilioNotifier(config)

# define four constants, one for the eye aspect ratio to indicate
# blink, a second and third constants to define baby sleeping and
# baby awake. The fourth constant is time between two alerts
EYE_AR_THRESH = config["eye_aspect_ratio_thresh"]
OPEN_THRESH_SECONDS = config["open_thresh_seconds"]
CLOSED_THRESH_SECONDS = config["closed_thresh_seconds"]
ALERT_TIMER_MINUTES = config["alert_timer_minutes"]

# boolean constant to display frames
VIDEO_ON = config["video"]

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
# uncomment the below if you use Rasberry Pi
# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=0).start()
time.sleep(1.0)

# signal trap to handle keyboard interrupt
signal.signal(signal.SIGINT, signal_handler)
print("[INFO] Press `ctrl + c` to exit, or 'q' to quit if you have" \
	" the display option on...")

# initilize previous alert sent time to few minutes ago
prev_alert_sent_time = datetime.now() - timedelta(minutes=ALERT_TIMER_MINUTES)
eyes_open_time = None
eyes_closed_time = None
is_sleeping = True

while True:
    
    # read frame and convert image to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    copy_frame = frame.copy()
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
        cv2.drawContours(copy_frame, [leftEyeHull], -1, (0, 255,0), 1)
        cv2.drawContours(copy_frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear >= EYE_AR_THRESH: # eyes open
                
            if not eyes_open_time:
                eyes_open_time = datetime.now()
            
            # calculate time to consider baby awake
            open_tresh = eyes_open_time + timedelta(seconds=OPEN_THRESH_SECONDS)
            
            if open_tresh <= datetime.now():
                
                eyes_closed_time = None
                
                # calculate next alert time
                next_alert_time = prev_alert_sent_time + timedelta(minutes=ALERT_TIMER_MINUTES)

                if next_alert_time <= datetime.now():
                    is_sleeping = False
                    eyes_open_time = None
                    
                    # send alert
                    dateAwake = date.today().strftime("%A, %B %d %Y")
                    msg = "Message from the baby on {} at {}: I'm awake!".format(dateAwake, datetime.now().strftime("%I:%M%p"))
                    send_alert(msg, frame)

                    prev_alert_sent_time = next_alert_time
                    
                    print("[INFO] alert sent: {}". format(msg))
                
                cv2.putText(copy_frame, "BABY IS AWAKE!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        else: #eyes closed
            if not eyes_closed_time:
                eyes_closed_time = datetime.now()

            # calculate time to consider baby slleepign
            closed_tresh = eyes_closed_time  + timedelta(seconds=CLOSED_THRESH_SECONDS)

            if closed_tresh <= datetime.now() and not is_sleeping:
                is_sleeping = True
                eyes_open_time = None
                prev_alert_sent_time = datetime.now() - timedelta(minutes=5)
                
                msg = "Message from the baby on {} at {}: Zzzzzzzzzzz (do not disturb)".format(dateAwake, datetime.now().strftime("%I:%M%p"))
                send_alert(msg, frame)

                print("[INFO] alert sent: {}". format(msg))
                
                cv2.putText(copy_frame, "Zzzzzzz!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # display eye aspect ratio
        cv2.putText(copy_frame, "EAR = {:.2f}".format(ear), (300,30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    if config["video"]:
        cv2.imshow('babymon', copy_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    time.sleep(1.0)

cv2.destroyAllWindows()
vs.stop()
