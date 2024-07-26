import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from typing import Tuple

@step
def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    select_features = ['protocol_type',
                        'service',
                        'flag',
                        'src_bytes',
                        'dst_bytes',
                        'count',
                        'same_srv_rate',
                        'diff_srv_rate',
                        'dst_host_srv_count',
                        'dst_host_same_srv_rate']
    X = X[select_features]
    try:
        logging.info("Starting data splitting.")
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info(f"Data splitting completed successfully. Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise e
