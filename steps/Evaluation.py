import logging
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RegressorMixin
from zenml import step
from typing import Tuple, Annotated
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import mlflow
# Define the Evaluation classes
class Evaluation(ABC):
    @abstractmethod
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e

class RMSE(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e

class Accuracy(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("Accuracy: {}".format(accuracy))
            return accuracy
        except Exception as e:
            logging.error("Error in calculating Accuracy: {}".format(e))
            raise e

class Precision(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Precision")
            precision = precision_score(y_true, y_pred)
            logging.info("Precision: {}".format(precision))
            return precision
        except Exception as e:
            logging.error("Error in calculating Precision: {}".format(e))
            raise e

class Recall(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Recall")
            recall = recall_score(y_true, y_pred)
            logging.info("Recall: {}".format(recall))
            return recall
        except Exception as e:
            logging.error("Error in calculating Recall: {}".format(e))
            raise e

class F1(Evaluation):
    def calculates_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1 Score")
            f1 = f1_score(y_true, y_pred)
            logging.info("F1 Score: {}".format(f1))
            return f1
        except Exception as e:
            logging.error("Error in calculating F1 Score: {}".format(e))
            raise e

@step
def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "primary_metric"], Annotated[float, "secondary_metric"]]:
    try:
        prediction = model.predict(X_test)
        
        if isinstance(model, RegressorMixin):
            mse_evaluator = MSE()
            mse = mse_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("MSE: {}".format(mse))
            
            r2_evaluator = R2()
            r2 = r2_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("R2 Score: {}".format(r2))
            
            rmse_evaluator = RMSE()
            rmse = rmse_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("RMSE: {}".format(rmse))
            
            return r2, rmse
        
        elif isinstance(model, (LogisticRegression, ClassifierMixin)):
            accuracy_evaluator = Accuracy()
            accuracy = accuracy_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("Accuracy: {}".format(accuracy))
            
            precision_evaluator = Precision()
            precision = precision_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("Precision: {}".format(precision))
            
            recall_evaluator = Recall()
            recall = recall_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("Recall: {}".format(recall))
            
            f1_evaluator = F1()
            f1 = f1_evaluator.calculates_scores(y_test.values, prediction)
            logging.info("F1 Score: {}".format(f1))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            return accuracy, f1
        
        else:
            raise ValueError("Unsupported model type: {}".format(type(model)))
        
    except Exception as e:
        logging.error("Error during model evaluation: {}".format(e))
        raise e
