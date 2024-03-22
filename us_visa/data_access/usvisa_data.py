from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USVisaException
from us_visa.logger import logging
from typing import Optional
import pandas as pd
import sys
import numpy as np


class USVisaData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise USVisaException(e, sys)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            logging.info(
                "Importing US Visa data from MongoDB in pandas Dataframe-Started"
            )

            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            df.replace(to_replace={"np": np.nan}, inplace=True)

            logging.info("Importing US Visa data from MongoDB is Completed")

            return df

        except Exception as e:
            raise USVisaException(e, sys)
