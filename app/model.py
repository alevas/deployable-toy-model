import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import configs
from app.utils.utils_classifier import ModelTypes


class ClassPredictor:
    def __init__(self,
                 model_type: str,
                 xg_booster: str = "dart",
                 svm_kernel: str = 'rbf',
                 svm_gamma: float = 0.3,
                 svm_C: float = 0.1,
                 pretrained_model_path: str = "") -> None:

        self.model_type = model_type

        # Choose from the implemented model types.
        if ModelTypes.has_value(self.model_type):
            if ModelTypes[model_type] == ModelTypes.XGBOOST:
                logging.info(f"Loading an XGBoost classifier...")
                self.model = XGBClassifier(booster=xg_booster)

            elif ModelTypes[self.model_type] == ModelTypes.MULTI_SVM:

                logging.info(f"Loading a Multiclass SVM...")
                self.model = svm.SVC(kernel=svm_kernel,
                                     gamma=svm_gamma,
                                     C=svm_C)

            elif ModelTypes[self.model_type] == ModelTypes.MLP:

                logging.info(f"Loading an MLP Neural Network...")
                self.model = MLPClassifier(random_state=42,
                                           max_iter=500,
                                           activation="logistic",
                                           learning_rate="adaptive",
                                           early_stopping=True)

            elif ModelTypes[self.model_type] == ModelTypes.PRETRAINED:
                logging.info(f"Initializing from a pretrained model...")
                self.load_model(
                    pretrained_model_path=pretrained_model_path)

        else:
            raise ValueError(f"Model type {self.model_type} not implemented!")

    def predict(self,
                X: np.ndarray,
                translated_return: bool = False,
                label_mapper: Optional[LabelEncoder] = None) -> np.ndarray:
        """
        Retrieve the predictions for a given input X.

        :param X: The input data.
        :param translated_return: Translate the numerical prediction values
                                    back to categorical
        :param label_mapper: The function that does the above.
        :return: The predictions for input X.
        """
        logging.info(f"Retrieving predictions")

        predictions = self.model.predict(X)

        if translated_return:
            predictions = label_mapper.inverse_transform(predictions)

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        A wrapper for invoking the classifier fit method, ignorant of the
        model's origin module.

        :param X: The input data.
        :param y: The corresponding label.
        """
        logging.info(f"Training the model...")
        self.model.fit(X, y)

    def save_model(self, pretrained_model_path: str = "") -> None:
        """
        Save the trained model.

        :param pretrained_model_path: The model save path.
        """
        save_path = os.path.join(configs.ROOT_DIR, pretrained_model_path,
                                 f'model_{self.model_type}.joblib')
        logging.info(f"Saving model under {save_path}...")
        joblib.dump(self.model, save_path)

    def load_model(self, pretrained_model_path: str = "") -> None:
        """
        Load the trained model.

        :param pretrained_model_path: The model save path.
        """
        logging.info(f"Loading model from {pretrained_model_path}...")
        save_path = os.path.join(configs.ROOT_DIR, pretrained_model_path)
        self.model = joblib.load(save_path)
