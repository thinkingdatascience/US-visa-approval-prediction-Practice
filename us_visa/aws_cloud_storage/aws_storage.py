from us_visa.aws_cloud_storage.aws_connection import s3Client
import sys
from us_visa.exception import USvisaException


# This is for Model Pusher
class SimpleStorageService:

    def __init__(self):
        s3_client = s3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [
                file_object for file_object in bucket.objects.filter(Prefix=s3_key)
            ]
            if len(file_objects) > 0:
                return True
            else:
                return False
        except Exception as e:
            raise USvisaException(e, sys)
