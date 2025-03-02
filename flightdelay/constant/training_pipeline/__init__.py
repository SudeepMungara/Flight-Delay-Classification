import os


"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "ARR_DELAY_CLS"
PIPELINE_NAME: str = "FlightDelayClassificationPipeline"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "preprocessed_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

"""
Columns to convert to categorical
"""
CATEGORICAL_COLUMNS = ['MONTH', 'DAY_OF_WEEK','DAY_OF_MONTH','UNIQUE_CARRIER',
            'NEW_ORIGIN', 'NEW_DEST', 'ORIGIN_CITY_TIME_ZONE', 'DEST_CITY_TIME_ZONE']

"""
Columns to exclude from the feature set
"""
EXCLUDE_COLUMNS=['ARR_DELAY_CLS', 'YEAR', 'QUARTER', 'FL_DATE', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR','DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEP_TIME', 'ARR_TIME',
       'ARR_DELAY', 'AIR_TIME','DISTANCE_GROUP','ARR_TIME_FORMAT_FLG', 'DEP_TIME_FORMAT_FLG', 'ARR_TIME_FRMT', 'DEP_TIME_FRMT', 'ARR_TIME_UTC', 'DEP_TIME_UTC', 'ARR_TIME_UTC_MINUTES', 'UTC_TIME_DIFFERENCE', 'DISTANCE_GROUP_CHK_FLG',
       'SCHD_ARR_TIME_FRMT', 'SCHD_ARR_UTC','DEPT_UTC_HR', 'SCHD_ARR_UTC_HR', 'ORIGIN', 'DEST', 'DEP_TIME_UTC_MINUTES']

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "FlightDelayData"
DATA_INGESTION_DATABASE_NAME: str = "FLIGHTDELAY"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2