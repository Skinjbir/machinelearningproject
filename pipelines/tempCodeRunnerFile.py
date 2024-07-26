from zenml.pipelines import pipeline
from steps.ingest_data import ingest_df
from steps.clean_and_encode_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    cleaned_df = clean_df(df)
    train_model(cleaned_df)
    evaluate_model(cleaned_df)
