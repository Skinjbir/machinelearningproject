import logging
import pandas as pd
from steps.clean_and_encode_data import clean_and_encode_data

def get_data_for_test() -> str:
    """Load, clean, and prepare a sample of data for testing."""
    try:
        # Load the dataset
        df = pd.read_csv("./data/Train_data.csv")
        
        # Sample a subset of data
        df = df.sample(n=100, random_state=42)  # Added random_state for reproducibility
        
        # Clean and encode the data
        df = clean_and_encode_data(df)  # Ensure this function returns the cleaned DataFrame
        
        # Drop the target column
        df.drop(["class"], axis=1, inplace=True)
        
        # Convert DataFrame to JSON format
        result = df.to_json(orient="split")
        
        return result
    except Exception as e:
        logging.error(f"An error occurred while processing the data: {e}")
        raise e
