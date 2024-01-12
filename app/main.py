import logging

import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from app.model import ClassPredictor
from app.utils import DataPreprocessor
from app.utils.utils_generic import load_model_preprocessor, \
    save_predictions, \
    ModelPreprocessor

description = "A very generic model, intended to predict anything*" \
              "given in a CSV format, exposed via FastAPI and deployable" \
              "with Docker."

app = FastAPI(title="Deployable Generic Model",
              description=description,
              summary="Given a CSV with data and a model type, \n"
                      "I create a CSV with predictions.",
              version="0.0.1",
              contact={
                  "name": "Alex Vasileiou ",
                  "email": "vasileiou.a@gmx.de",
              }, )


@app.get("/")
def status() -> str:
    """
    Verify the status of the service.

    :return: An assurance that the service is running.
    """
    return "API is working fine"


@app.post("/predict")
def predict(payload: dict,
            outfile_path: str = "input_data_with_predictions.csv") -> str:
    """
    After loading a pretrained model and data preprocessor, save the
    predictions as an extra column to the CSV input file.

    :param payload: The received payload. Expecting csv_input_path and
                    pretrained_model_path keys, outfile_path as optional.
    :param outfile_path: The destination path of the dataframe.
    :return: A verification that the predictions have been generated and saved.
    """

    csv_path = payload["csv_input_path"]
    pretrained_model_path = payload["pretrained_model_path"]

    if "outfile_path" in payload.keys():
        outfile_path = payload["outfile_path"]

    if ModelPreprocessor.model is None or ModelPreprocessor.preprocessor is None:
        logging.debug("Initializing the model and the data preprocessor...")
        ModelPreprocessor.model, ModelPreprocessor.preprocessor = load_model_preprocessor(
            csv_path=csv_path,
            pretrained_model_path=pretrained_model_path)
        ModelPreprocessor.preprocessor.preprocess_data()

    ModelPreprocessor.preprocessor.df = pd.read_csv(csv_path)
    ModelPreprocessor.preprocessor.preprocess_data()

    X, _ = ModelPreprocessor.preprocessor.get_data_split(split_sets=False)

    logging.debug("Retrieving predictions...")
    predictions = ModelPreprocessor.model.predict(X, translated_return=True,
                                                  label_mapper=ModelPreprocessor.preprocessor.le)
    return save_predictions(predictions, outfile_path)


@app.post("/train_predict")
def train_predict(payload: dict,
                  outfile_path: str = "train_data_with_predictions.csv") -> str:
    """
    Train a model and a preprocessor, retrieve accuracy scores, save the
    predictions as an extra column to the CSV input file.

    :param payload: The received payload. Expecting csv_input_path,
                    outfile_path as optional.
    :param outfile_path: The destination path of the dataframe.
    :return: A verification that the predictions have been generated and saved
    """

    # Data preprocessing
    csv_train = payload["csv_train"]

    ModelPreprocessor.preprocessor = DataPreprocessor(train_set_path=csv_train,
                                                     test_set_path=payload["csv_test"] if "csv_test" in payload else '',
                                                      label_col_name=payload["label_col_name"])
    ModelPreprocessor.preprocessor.preprocess_data()

    X_train, X_test, y_train, y_test = ModelPreprocessor.preprocessor.get_data_split()

    # Model train and predictions
    ModelPreprocessor.model = ClassPredictor(model_type="XGBOOST")
    ModelPreprocessor.model.fit(X_train, y_train)

    predict_train = ModelPreprocessor.model.predict(X_train)
    predict_test = ModelPreprocessor.model.predict(X_test)

    accuracy_train = accuracy_score(y_train, predict_train)
    accuracy_test = accuracy_score(y_test, predict_test)

    X, _ = ModelPreprocessor.preprocessor.get_data_split(split_sets=False)
    predictions = ModelPreprocessor.model.predict(X, translated_return=True,
                                                  label_mapper=ModelPreprocessor.preprocessor.le)

    # Save things
    save_predictions(predictions, outfile_path)
    ModelPreprocessor.model.save_model()
    ModelPreprocessor.preprocessor.save_preprocessor(destroy_df=False)


    # Confusion matrix for the XGBoost model
    confusion_matrix_ = confusion_matrix(y_test, predict_test)
    confusion_matrix_display_ = ConfusionMatrixDisplay(confusion_matrix_,
                                                       display_labels=ModelPreprocessor.preprocessor.le.classes_)

    confusion_matrix_display_.plot(cmap="Oranges").figure_.savefig("confusion_matrix.jpg")

    return f"Accuracy on the train set: {accuracy_train} \n" \
           f"Accuracy on the test set: {accuracy_test}"


if __name__ == "__main__":
    model_preprocessor = ModelPreprocessor
    uvicorn.run(app, host='127.0.0.1', port=8000)
