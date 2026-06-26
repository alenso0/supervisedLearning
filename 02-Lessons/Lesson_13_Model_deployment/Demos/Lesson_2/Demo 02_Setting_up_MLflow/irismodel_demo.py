import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the Iris dataset
data = load_iris(as_frame=True)
X = data.data
y = data.target

# EDA: Summary statistics and visualization
eda_summary = X.describe()

# Pairplot for visualizing relationships between features
sns.set(style="ticks")
eda_pairplot = sns.pairplot(data=eda_summary, diag_kind="kde")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Generate a classification report
classification_rep = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)

# Start an MLflow run
with mlflow.start_run():

    # Log EDA summary as a text artifact
    eda_summary_file = "eda_summary.txt"
    with open(eda_summary_file, "w") as eda_file:
        eda_file.write(eda_summary.to_string())
    mlflow.log_artifact(eda_summary_file)

    # Log the EDA pairplot as an image artifact
    pairplot_file = "eda_pairplot.png"
    eda_pairplot.savefig(pairplot_file)
    mlflow.log_artifact(pairplot_file)

    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "random_state": 42
    })

    # Log metrics (precision, recall, f1-score)
    for class_name in data.target_names:
        metrics = classification_rep[class_name]
        mlflow.log_metrics({
            f"precision_{class_name}": metrics['precision'],
            f"recall_{class_name}": metrics['recall'],
            f"f1-score_{class_name}": metrics['f1-score']
        })

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Log the classification report as a text artifact
    classification_report_file = "classification_report.txt"
    with open(classification_report_file, "w") as text_file:
        text_file.write(classification_report(y_test, y_pred, target_names=data.target_names))
    mlflow.log_artifact(classification_report_file)
