import logging
import os
from typing import Tuple

import numpy as np

import configs
from app.model import ClassPredictor
from app.utils.data_preprocessor import DataPreprocessor


class ModelPreprocessor:
    model: ClassPredictor = None
    preprocessor: DataPreprocessor = None


def load_model_preprocessor(csv_path: str, pretrained_model_path: str) -> \
        Tuple[ClassPredictor, DataPreprocessor]:
    """
    Load a pretrained model and a custom saved preprocessor.
    We need the preprocessor because of the categorical-numerical encoding,
    which might result in different values if not consistently used.

    :param csv_path: The path to the input CSV.
    :param pretrained_model_path: The pretrained model path.
    :return: Tuple(ClassPredictor, DataPreprocessor)
    """
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(
        pretrained_preprocessor_path="preprocessor.joblib", data_path=csv_path)

    model = ClassPredictor(model_type="PRETRAINED",
                           pretrained_model_path=pretrained_model_path)
    return model, preprocessor


def save_predictions(predictions: np.ndarray, outfile_path: str) -> str:
    """
    Appends the predictions to the input DataFrame and saves the file under
    data/processed/<outfile_path>.csv.

    :param predictions: A numpy array with the model predictions.
    :param outfile_path: The destination of the final CSV file.
    """
    logging.debug("Saving the dataframe with predictions...")
    ModelPreprocessor.preprocessor.df["predictions"] = predictions
    ModelPreprocessor.preprocessor.df.to_csv(
        path_or_buf=os.path.join(configs.ROOT_DIR, "data", "processed",
                                 outfile_path), escapechar="^")
    return f"Saved predictions as a new column in file {outfile_path}"
