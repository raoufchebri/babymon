# import the necessary packages
from twilio.rest import Client
import boto3
from threading import Thread

class TwilioNotifier:
    def __init__(self, conf):
        self.conf = conf

    def send(self, msg, tempVideo):
        t = Thread(target=self._send, args=(msg, tempVideo,))
        t.start()

    def _send(self, msg, tempVideo):
        s3 = boto3.client("s3",
            aws_access_key_id=self.conf["aws_access_key_id"],
            aws_secret_access_key=self.conf["aws_secret_access_key"],
        )

        filename = tempVideo.path[tempVideo.path.rfind("/") + 1:]
        s3.upload_file(tempVideo.path, self.conf["s3_bucket"],
            filename, ExtraArgs={"ACL": "public-read", 
            "ContentType": "video/mp4"})

        location = s3.get_bucket_location(
			Bucket=self.conf["s3_bucket"])["LocationConstraint"]
        url = "https://s3-{}.amazonaws.com/{}/{}".format(location,
			self.conf["s3_bucket"], filename)

        client = Client(self.conf["twilio_sid"],
			self.conf["twilio_auth"])

        client.messages.create(to=self.conf["twilio_to"], 
			from_=self.conf["twilio_from"], body=msg
            , media_url=url
            )

        tempVideo.cleanup()