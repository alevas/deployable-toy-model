import os
import unittest

import configs
from app import ClassPredictor


class TestModelSaveLoad(unittest.TestCase):
    def test_model_save_load(self):
        model = ClassPredictor(model_type="XGBOOST")
        model.save_model(pretrained_model_path="model_XGBOOST_TEST.joblib")
        del model
        model = ClassPredictor(model_type="PRETRAINED",
                               pretrained_model_path="model_XGBOOST_TEST.joblib")
        self.assertIsInstance(model, ClassPredictor, "Model save-load failed!")
        os.remove(os.path.join(configs.ROOT_DIR, "model_XGBOOST_TEST.joblib"))


if __name__ == '__main__':
    unittest.main()
