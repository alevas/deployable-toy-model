import unittest

from app.utils import DataPreprocessor


class TestModelSaveLoad(unittest.TestCase):
    def test_model_save_load(self):
        data_preprocessor = DataPreprocessor(
            train_set_path="data/raw/data.csv")
        data_preprocessor.save_preprocessor()
        del data_preprocessor
        data_preprocessor = DataPreprocessor()
        data_preprocessor.load_preprocessor(
            pretrained_preprocessor_path="preprocessor.joblib",
            data_path="data/raw/data.csv")
        self.assertIsInstance(data_preprocessor, DataPreprocessor,
                              "Model save-load failed!")


if __name__ == '__main__':
    unittest.main()
