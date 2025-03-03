from flightdelay.components.data_ingestion import DataIngestion
from flightdelay.components.data_validation import DataValidation
from flightdelay.components.data_transformation import DataTransformation
from flightdelay.logging.logger import logging
from flightdelay.exception.exception import FlighDelayException
from flightdelay.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig

import sys


if __name__ == "__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Data Ingestion started")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the Data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")

        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("data Transformation completed")


    except Exception as e:
        raise FlighDelayException(e,sys)
