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
Columns to preprocess
"""
PREPROCESS_NUMERIC_FEATURES = ['DISTANCE','EXPECTED_DURATION']
PREPROCESS_ORDINAL_FEATURES = ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK'] #Features with ordinal characterstics
PREPROCESS_CATEGORICAL_FEATURES = ['UNIQUE_CARRIER','ORIGIN_CITY_TIME_ZONE', 'DEST_CITY_TIME_ZONE','NEW_ORIGIN', 'NEW_DEST']# Features where one-hot encoding is required
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "FlightDelayData"
DATA_INGESTION_DATABASE_NAME: str = "FLIGHTDELAY"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""
Model Trainer related constant start with MODEL TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME = "FlightDelay"