from flightdelay.logging import logger
from flightdelay.exception.exception import FlighDelayException
from flightdelay.entity.config_entity import DataIngestionConfig
from flightdelay.entity.artifact_entity import DataIngestionArtifact

import os
import certifi
import ssl
import sys
import pandas as pd
import pymongo
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

ca = certifi.where()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            return df
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        """
        Export data into feature store
        """
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def group_rare_categories_series(self,series, threshold):
        """
        Groups rare categories as others using threshold
        """
        try:
            # Compute the frequency counts for the series
            counts = series.value_counts(normalize=True)
            # Replace categories that occur less than threshold times with 'Other'
            return series.apply(lambda x: x if counts[x] >= threshold else 'Other')
        except Exception as e:
            raise FlighDelayException(e,sys)
    def get_outlier_datapoints(self,data,col_name):
        """
        Identifies outlier datapoints in a specified column of a DataFrame using the Interquartile Range (IQR) method.
        """
        Q1 = data[col_name].quantile(0.25)
        Q3 = data[col_name].quantile(0.75)
        IQR = Q3-Q1
        lower_bound = Q1-1.5*IQR
        upper_bound = Q3+1.5*IQR
        lower_bound_outlier_datapoints = data[(data[col_name]<lower_bound)]
        upper_bound_outlier_datapoints = data[(data[col_name]>upper_bound)]
        return lower_bound_outlier_datapoints,upper_bound_outlier_datapoints, lower_bound, upper_bound
            
    def feature_engineering(self,dataframe: pd.DataFrame):
        try:

            air_time_lower_outlier, air_time_upper_outlier, air_time_lower_bound, air_time_upper_bound = self.get_outlier_datapoints(dataframe,'AIR_TIME')
            distance_lower_outlier, distance_upper_outlier, distance_lower_bound, distance_upper_bound = self.get_outlier_datapoints(dataframe,'DISTANCE')
            set1 = set(air_time_upper_outlier.apply(tuple, axis=1))
            set2 = set(distance_upper_outlier.apply(tuple, axis=1))
            sym_diff = set1.symmetric_difference(set2)
            df_diff = pd.DataFrame(list(sym_diff), columns=air_time_upper_outlier.columns)
            # dropping non-overlapping outliers
            df_main_tuples = dataframe.apply(tuple, axis=1)
            df_outlier_tuples = set(df_diff.apply(tuple, axis=1))
            # Keep only rows that are NOT in df_outlier
            dataframe = dataframe[~df_main_tuples.isin(df_outlier_tuples)]
            
            dataframe['EXPECTED_DURATION']=(dataframe['SCHD_ARR_TIME_UTC_MINUTES']-dataframe['DEP_TIME_UTC_MINUTES'])+1
            dataframe['ARR_DELAY_CLS'] = dataframe['ARR_DELAY_CLS'].map({'On-Time': 0, 'Delay': 1})
            dataframe['NEW_ORIGIN'] = self.group_rare_categories_series(dataframe['ORIGIN'], threshold=0.01)
            dataframe['NEW_DEST'] = self.group_rare_categories_series(dataframe['DEST'], threshold=0.01)
            # for col in self.data_ingestion_config.categorical_columns:
            #     dataframe[col] = dataframe[col].astype('category')
            return dataframe
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            X = dataframe.loc[:, ~dataframe.columns.isin(self.data_ingestion_config.exclude_columns)]
            y = dataframe['ARR_DELAY_CLS']
            X_train, X_test, y_train, y_test = train_test_split(
                X,y, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logger.logging.info("Performed train test split on the dataframe")

            logger.logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logger.logging.info(f"Exporting train and test file path.")

            train_data = pd.concat([X_train, y_train], axis=1)

            train_data.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_data = pd.concat([X_test, y_test], axis=1)

            test_data.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logger.logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = (
                self.export_collection_as_dataframe()
                .pipe(self.export_data_into_feature_store)
                .pipe(self.feature_engineering))
            self.split_data_as_train_test(dataframe)

            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise FlighDelayException