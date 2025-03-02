from flightdelay.components.data_ingestion import DataIngestion
from flightdelay.logging.logger import logging
from flightdelay.exception.exception import FlighDelayException
from flightdelay.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
import sys


if __name__ == "__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Data Ingestion started")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise FlighDelayException(e,sys)
