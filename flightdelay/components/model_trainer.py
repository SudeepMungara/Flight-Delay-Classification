from flightdelay.exception.exception import FlighDelayException 
from flightdelay.logging.logger import logging
from flightdelay.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from flightdelay.entity.config_entity import ModelTrainerConfig

from flightdelay.utils.model_utils.model.estimator import FlightDelayModel
from flightdelay.utils.main_utils.utils import save_object,load_object
from flightdelay.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from flightdelay.utils.model_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import os,sys
import mlflow
import dagshub
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()
dagshub.init(repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), repo_name='Flight-Delay-Classification', mlflow=True)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise FlighDelayException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric,prefix):
        
        mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_params(best_model.get_params())
            mlflow.log_metric(f"{prefix}_f1_score", classificationmetric.f1_score)
            mlflow.log_metric(f"{prefix}_precision", classificationmetric.precision_score)
            mlflow.log_metric(f"{prefix}_recall", classificationmetric.recall_score)
            mlflow.sklearn.log_model(best_model,"model")

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="best_model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
                
    def train_model(self,X_train,y_train,x_test,y_test):
        on_time_cls = np.sum(y_train == 0)
        delay_cls = np.sum(y_train == 1)
        scale_pos_weight = on_time_cls / delay_cls
        models = {
            'Logistic Regression': LogisticRegression(random_state=42,class_weight='balanced',penalty='l2'), #using L2 penalty to handle multicollinearity
            'Random Forest': RandomForestClassifier(random_state=42,class_weight='balanced'),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42,scale_pos_weight=scale_pos_weight, enable_categorical=False)}
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'solver':['liblinear']

            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0]
            }}
        model_report,best_estimator =evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                        models=models,param=param_grids)
        # best_model_score = max(sorted(model_report.values()))

        # best_model_name = list(model_report.keys())[
        #     list(model_report.values()).index(best_model_score)
        # ]
        # best_model = models[best_model_name]
        # # Train
        # y_train_pred=best_model.predict(X_train)
        # classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        # self.track_mlflow(best_model,classification_train_metric,prefix="train")

        # # Test
        # y_test_pred=best_model.predict(x_test)
        # classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        # self.track_mlflow(best_model,classification_test_metric,prefix="test")
        
        for _model in model_report.keys():
            best_model = model_report[_model]['best_model']
            y_test_pred=best_model.predict(x_test)
            classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
            if best_model == best_estimator:
                self.track_mlflow(best_model,classification_test_metric,prefix="best_model")
                y_train_pred=best_model.predict(X_train)
                classification_train_metric_bm=get_classification_score(y_true=y_train,y_pred=y_train_pred)
                classification_test_metric_bm=get_classification_score(y_true=y_test,y_pred=y_test_pred)
            else:
                self.track_mlflow(best_model,classification_test_metric,prefix="test_model")
            
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        FlightDelay_model=FlightDelayModel(preprocessor=preprocessor,model=best_estimator)
        save_object(self.model_trainer_config.trained_model_file_path,obj=FlightDelay_model)
        save_object("final_model/model.pkl",best_estimator)

        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                            train_metric_artifact=classification_train_metric_bm,
                            test_metric_artifact=classification_test_metric_bm
                            )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
            
        except Exception as e:
            raise FlighDelayException(e,sys)