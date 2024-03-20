import pymongo
import certifi

ca = certifi.where()

import os
import sys

from us_visa.logger import logging
from us_visa.exception import USVisaException
from us_visa.constants import DATABASE_NAME, MONGODB_URL_KEY


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            logging.info("Attempting MongoDB connection")
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment Name: {MONGODB_URL_KEY} is not set")

                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                self.client = MongoDBClient.client
                self.database_name = database_name
                self.database = self.client[database_name]
                logging.info("MongoDB connection successfull")

        except Exception as e:
            raise USVisaException(e, sys)
