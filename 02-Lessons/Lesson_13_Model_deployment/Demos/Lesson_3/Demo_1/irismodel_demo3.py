import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris(as_frame=True)
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a function to train and log metrics
def train_and_log_metrics(n_estimators, max_depth):
    # Create and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and generate a classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Log metrics to MLflow
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log the classification report as an artifact
        with open("classification_report.txt", "w") as text_file:
            text_file.write(classification_rep)
        mlflow.log_artifact("classification_report.txt")

        # Visualize metrics (accuracy)
        plt.figure()
        plt.title("Accuracy Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.plot([accuracy])
        plt.savefig("accuracy_plot.png")
        mlflow.log_artifact("accuracy_plot.png")
        

# Continuously monitor and adjust the pipeline
for iteration in range(5):
    n_estimators = 100 + iteration * 50
    max_depth = 5 + iteration
    train_and_log_metrics(n_estimators, max_depth)
