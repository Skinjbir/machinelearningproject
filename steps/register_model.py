from zenml.steps import Output
from zenml import step
from sklearn.linear_model import LogisticRegression


@step
def register_model(model: LogisticRegression) -> LogisticRegression:
    return model
