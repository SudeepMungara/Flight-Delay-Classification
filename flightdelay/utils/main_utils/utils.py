from flightdelay.logging.logger import logging
import numpy as np
import pickle
from flightdelay.exception.exception import FlighDelayException
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import f1_score
import os,sys
import yaml

def read_yaml_file(file_path: str) -> dict:
    """
    read data from yaml file
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FlighDelayException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    write data to yaml file
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise FlighDelayException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FlighDelayException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object saved")
    except Exception as e:
        raise FlighDelayException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise FlighDelayException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FlighDelayException(e, sys) from e
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    # try:
    #     report = {}

    #     for i in range(len(list(models))):
    #         model = list(models.values())[i]
    #         para=param[list(models.keys())[i]]

    #         gs = GridSearchCV(model,para,cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
    #         gs.fit(X_train,y_train)

    #         model.set_params(**gs.best_params_)
    #         model.fit(X_train,y_train)

    #         y_train_pred = model.predict(X_train)

    #         y_test_pred = model.predict(X_test)

    #         train_model_score = f1_score(y_train, y_train_pred)

    #         test_model_score = f1_score(y_test, y_test_pred)

    #         report[list(models.keys())[i]] = test_model_score
    #     return report

    # except Exception as e:
    #     raise FlighDelayException(e, sys)
    try:
        report = {}
        best_model_score = 0
        best_model = None
        for model_name,model in models.items():
            
            grid_search = GridSearchCV(model,param[model_name],cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1,refit=True)
            grid_search.fit(X_train, y_train)
            if model_name not in report:
                report[model_name] = {}
            report[model_name]['best_params']  = grid_search.best_params_
            report[model_name]['best_model']  = grid_search.best_estimator_
            y_pred = grid_search.predict(X_test)
            test_model_score = f1_score(y_test, y_pred)
            report[model_name]['model_score'] = test_model_score
            if test_model_score>best_model_score:
                best_model_score = test_model_score
                best_model = grid_search.best_estimator_
        return report,best_model
    except Exception as e:
        raise FlighDelayException(e, sys)