# Babymon
Simple baby monitor that detects facial landmarks and eye aspect ratio then sends notification when baby is awake to has just slept

# Usage
- Download a trained facial shape predictor from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" and add it to the ./models directory
- Provide AWS access information and S3 Bucket on config.json:
    - Create an ACCOUNT account on https://aws.amazon.com    
    - On IAM, create a new Group and name it for example "s3pi"
    - In your new group permissions, Attach Policy "AmazonS3FullAccess"
    - Create a new user and add it to the group
    - Copy Access key ID and Secret Access Key
    - Create an S3 bucket and make sure that your user has access to the bucket
- Provide Twilio information on config.json
    - Create a Twilio account on https://twilio.com
    - Copy ACCOUNT SID, AUTH Token and NUMBER
- Execute the python file using: <pre>python babymon.py --config config/config.json</pre>

# Credit
This project is derived from a blog posts by Adrian Rosebrock on pyimagesearch
Drowsiness detection with OpenCV https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/
