# model.py
import numpy as np
import mlflow
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("http://<MLFLOW_SERVER_IP>:5000")  # Replace with your MLflow server IP
mlflow.set_experiment("my_regression_experiment")

def train_model():
    mlflow.start_run()
    
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log metrics and parameters
    mlflow.log_params({"random_state": 0, "test_size": 0.2})
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred), "r2": r2_score(y_test, y_pred)})
    
    # Save the model
    mlflow.sklearn.log_model(model, "model")
    
    mlflow.end_run()

if __name__ == "__main__":
    train_model()
