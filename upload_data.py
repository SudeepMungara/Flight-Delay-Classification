import os
import sys
import json
import certifi
import pymongo
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where() # to have trusted certificates

from flightdelay.logging import logger
from flightdelay.exception.exception import FlighDelayException

class FlightDelayExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def csv_to_json(self,csv_file_path:str):
        try:
            df = pd.read_csv(csv_file_path)
            df.reset_index(inplace=True,drop=True)
            records=list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise FlighDelayException(e,sys)


if __name__ == "__main__":
    try:
        FILE_PATH="/Users/sudeepmungara/Documents/Personal_Projects/Flight-Delay-Classification/dataset/Flight Delay Dataset.csv"
        DATABASE="FLIGHTDELAY"
        Collection="FlightDelayData"
        flight_delay_extract = FlightDelayExtract()
        print(flight_delay_extract)
        json_data = flight_delay_extract.csv_to_json(FILE_PATH)
        logger.logging.info(f"Data extracted successfully")
        no_of_records=flight_delay_extract.insert_data_mongodb(json_data,DATABASE,Collection)
        print(no_of_records)
        logger.logging.info(f"Number of inserted records: {no_of_records}")
    except FlighDelayException as e:
        logger.logging.error(f"Error in extracting data: {e}")

