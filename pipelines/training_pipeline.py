from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_and_encode_data import clean_and_encode_data
from steps.split_data import split_data
from steps.train_model import train_model
from steps.evaluation_model import evaluate_model
from steps.register_model import register_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    X, y = clean_and_encode_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    register_model(model)
    evaluate_model(model, X_test, y_test)


