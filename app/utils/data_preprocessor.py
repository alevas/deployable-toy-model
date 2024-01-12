import logging
import os
from dataclasses import dataclass
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import configs


@dataclass
class DataPreprocessor:

    def __init__(self,
                 train_set_path="data/raw/moviesdataset_2023.csv",
                 test_set_path='',
                 label_col_name="genre"):
        self.train_set_path = os.path.join(configs.ROOT_DIR, train_set_path)
        if test_set_path:
            self.test_set_path = os.path.join(configs.ROOT_DIR, train_set_path)
        else:
            self.test_set_path = ''
        self.df = pd.read_csv(self.train_set_path)

        self.enc = OrdinalEncoder()
        self.le = LabelEncoder()

        self.label_col_name = label_col_name
        self.X, self.y = None, None

    def preprocess_data(self) -> None:
        """
        Preprocess the input DataFrame and split into arrays X(features) and
        y(labels).
        """
        if self.df is None:
            self.df = pd.read_csv(
                os.path.join(configs.ROOT_DIR, self.train_set_path))
        if self.test_set_path:
            self.df = pd.read_csv(
                os.path.join(configs.ROOT_DIR, self.test_set_path))
        else:
            logging.info("Test set path not provided, splitting from train.")
            self.X, self.y = self._split_to_x_y()
        self.X = np.nan_to_num(self.X)

    def get_data_split(self,
                       split_sets: bool = True,
                       test_size=0.2,
                       random_state=42) -> Union[Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieve the data split of X and y.

        :param split_sets: Split dataset into Train and Test sets.
        :param test_size: The ration of Test to Train sets.
        :param random_state: For reproducibility.
        :return:
        """
        if split_sets:
            return train_test_split(self.X, self.y,
                                    test_size=test_size,
                                    random_state=random_state)
        else:
            return self.X, self.y

    def _split_to_x_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform categorical data and labels into numeric. For further
        info, look into the scikit-learn implementations of LabelEncoder and
        OrdinalEncoder.

        SPOILER: They are the same, I tested it. The former works on n*1 and
        the later on n*m arrays

        :return: The transformed X data and y labels
        """
        X = self.enc.fit_transform(self.df.drop(self.label_col_name, axis=1))

        # The LabelEncoder turns categorical to numerical labels (m*1 vector)

        y = self.le.fit_transform(self.df[self.label_col_name])

        return X, y

    def save_preprocessor(self, pretrained_preprocessor_path: str = "",
                          destroy_df: bool = True) -> None:
        """
        Save the preprocessor.

        :param pretrained_preprocessor_path:
        :param destroy_df:
        """
        save_path = os.path.join(configs.ROOT_DIR,
                                 pretrained_preprocessor_path,
                                 f'preprocessor.joblib')
        logging.info(f"Saving model under {save_path}...")
        if destroy_df:
            # for memory and consistency reasons
            self.df = None
        joblib.dump(self, save_path)

    def load_preprocessor(self, pretrained_preprocessor_path,
                          data_path) -> None:
        """
        Load a saved preprocessor.

        :param pretrained_preprocessor_path: The preprocessor save path.
        :param data_path: The dataframe path.
        :return:
        """
        logging.info(f"Loading model from {pretrained_preprocessor_path}...")
        load_path = os.path.join(configs.ROOT_DIR,
                                 pretrained_preprocessor_path)
        preprocessor = joblib.load(load_path)
        preprocessor.train_set_path = data_path

        self.enc = preprocessor.enc
        self.le = preprocessor.le

        self.label_col_name = preprocessor.label_col_name
