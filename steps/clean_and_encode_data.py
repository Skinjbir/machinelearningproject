import logging
from typing import Tuple
import pandas as pd
from zenml import step
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import joblib
# Save encoders



@step
def clean_and_encode_data(df: pd.DataFrame, scale_method: str = 'standardize') -> Tuple[pd.DataFrame, pd.Series]:
    try:
        logging.info("Starting data cleaning and encoding.")
        
        # Drop duplicate rows
        initial_shape = df.shape
        df = df.drop_duplicates()
        logging.info(f"Removed duplicates: {initial_shape} -> {df.shape}")
        
        # Initialize the LabelEncoder
        label_encoders = {}

        # Encode categorical columns' values (not column labels)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            label_encoders[col] = label_encoder
            logging.info(f"Encoded column '{col}' with labels: {label_encoder.classes_}")

        # Encode the target column
        label_encoder_target = LabelEncoder()
        df['class'] = label_encoder_target.fit_transform(df['class'])
        logging.info(f"Encoded target column 'class' with labels: {label_encoder_target.classes_}")


        joblib.dump(label_encoders, 'saved_model/label_encoders.pkl')
        joblib.dump(label_encoder_target, 'saved_model/label_encoder_target.pkl')

        # Handle missing values
        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.mean(numeric_only=True))
            logging.info("Filled missing values with column means.")
        
        # Separate features and target variable
        X = df.drop(columns=['class'])
        y = df['class']

        # Store column names
        column_names = X.columns
        
        # Scale features
        if scale_method == 'standardize':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            logging.info("Standardized features.")
        elif scale_method == 'normalize':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            logging.info("Normalized features.")
        else:
            raise ValueError("Invalid scale_method. Choose 'standardize' or 'normalize'.")
        
        # Convert to DataFrame and restore column names
        X = pd.DataFrame(X, columns=column_names)

        return X, y
    except Exception as e:
        logging.error(f"Error during data cleaning and encoding: {e}")
        raise e
