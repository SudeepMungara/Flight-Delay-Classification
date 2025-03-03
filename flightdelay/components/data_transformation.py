import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,OrdinalEncoder

from flightdelay.constant.training_pipeline import TARGET_COLUMN

from flightdelay.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from flightdelay.entity.config_entity import DataTransformationConfig
from flightdelay.exception.exception import FlighDelayException 
from flightdelay.logging.logger import logging
from flightdelay.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FlighDelayException(e, sys)
        
    
    def get_preprocessor_pipeline(self):
        """ Function to create scikit-learn pipeline

        Args:
            classifier_model: model
            preprocessor: preprocessor pipeline

        Returns:
            pipeline: pipeline with preprocessor and model steps
        """
        try:
            preprocessor = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(), self.data_transformation_config.preprocess_ordinal_features),
                ('cat', OneHotEncoder(), self.data_transformation_config.preprocess_categorical_features ),
                ('num',MinMaxScaler(),self.data_transformation_config.preprocess_numeric_features),
            ])
            pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)])
            return pipeline
        except Exception as e:
            raise FlighDelayException(e, sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]


            preprocessor=self.get_preprocessor_pipeline()

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
            
            
            train_arr = np.c_[transformed_input_train_feature.toarray(), np.array(target_feature_train_df).reshape(-1, 1) ]
            test_arr = np.c_[ transformed_input_test_feature.toarray(), np.array(target_feature_test_df).reshape(-1, 1) ]

            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            save_object("final_model/preprocessor.pkl", preprocessor_object)

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
            
        except Exception as e:
            raise FlighDelayException(e,sys)