from enum import Enum


# This enum allows us to safeguard the string arguments and validate that
# the user is trying to load an implemented model.


class ModelTypes(Enum):
    XGBOOST = "XGBOOST"
    MULTI_SVM = "MULTI_SVM"
    MLP = "MLP"
    PRETRAINED = "PRETRAINED"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
