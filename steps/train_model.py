import joblib
from sklearn.linear_model import LogisticRegression
from zenml import step
import pandas as pd

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    try:
        # Initialize the model
        model = LogisticRegression(max_iter=10000, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Save the model for future use
        joblib.dump(model, './saved_model/model.h')
        
        # Return the model object directly
        return model
    except Exception as e:
        # Raise the exception to be handled upstream
        raise e
