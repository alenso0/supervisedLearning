# backend/backend_models.py
# All backend code: loading, processing, five algorithms, metrics, comparison.

import pandas as pd
import numpy as np
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# --------------- Data loading ---------------

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load CSV/Excel dataset from local path.
    The file must be provided by the user; no preloaded data.
    """
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    elif path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

# --------------- Processing (shared for all POCs) ---------------

def process_dataset(df: pd.DataFrame, target_col: str = "long_stay_label"):
    """
    Full processing pipeline BEFORE EDA and BEFORE training.
    """

    # 1. Map real columns to internal schema
    column_map = {
        "eid": "patient_id",
        "lengthofstay": "length_of_stay"
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # 2. Create binary target from length_of_stay if not present
    if target_col not in df.columns:
        if "length_of_stay" not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found and 'length_of_stay' "
                "is missing. Please ensure the dataset has 'lengthofstay'."
            )
        # Long stay rule: LOS >= 4 days -> 1, else 0 (adjustable)
        df[target_col] = (df["length_of_stay"] >= 4).astype(int)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # 3. Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].mode().empty:
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 4. Outlier handling: IQR clipping
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        df[col] = df[col].clip(low, high)

    # 5. Split into X and y
    y = df[target_col]

    # Columns that should NOT be used as features (to avoid leakage / IDs)

    leak_or_id_cols = ["length_of_stay", "eid", "vdate", "facid"]
    cols_to_drop = [target_col] + [c for c in leak_or_id_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # sanity check – avoid single-class target
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            f"Target has only one class after thresholding. "
            f"Counts: {class_counts.to_dict()}. "
            f"Try a different LOS threshold in process_dataset."
        )

    # 6. Encode categoricals
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_enc = X.copy()
    label_encoders = {}

    # Label encode binary categoricals
    for col in cat_cols:
        if X_enc[col].nunique() == 2:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col])
            label_encoders[col] = le

    # One-hot encode remaining categoricals
    multi_cat = [c for c in cat_cols if c not in label_encoders]
    if multi_cat:
        X_enc = pd.get_dummies(X_enc, columns=multi_cat, drop_first=True)

    return X_enc, y, label_encoders

# --------------- Train/test split ---------------

def create_train_test_splits(X, y, test_size=0.3, random_state=42):
    """
    Create train/test splits (default 70:30).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --------------- Single algorithm POC helper ---------------

def train_and_evaluate_model(model_name, estimator, X_train, X_test, y_train, y_test):
    """
    Single POC: wrap model in a Pipeline (StandardScaler + model),
    fit, and compute all required metrics.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])

    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    # probabilities for ROC/AUC
    if hasattr(pipe, "predict_proba"):
        y_test_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        if hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_test)
            y_test_proba = 1 / (1 + np.exp(-scores))
        else:
            y_test_proba = y_test_pred.astype(float)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    rep = classification_report(y_test, y_test_pred, output_dict=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr = thresholds[opt_idx]

    return {
        "name": model_name,
        "pipeline": pipe,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cm": cm,
        "report": rep,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "opt_thr": opt_thr
    }

# --------------- Train all 5 algorithms (5 POCs) ---------------

def train_all_algorithms(X_train, X_test, y_train, y_test):
    """
    Train all 5 algorithms and return a dict of results.
    Lighter / faster model configurations.
    """
    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,               # try 4–8
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42

        ),
        # Linear SVM without probability=True for speed
        "SVM": SVC(
            kernel="linear",
            probability=False,
            random_state=42
        )
    }

    results = {}
    for name, est in algorithms.items():
        results[name] = train_and_evaluate_model(
            model_name=name,
            estimator=est,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
    return results

# --------------- Build comparison DataFrame ---------------

def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a summary table (DataFrame) of metrics for all algorithms.
    """
    rows = []
    for name, res in results.items():
        rep = res["report"]
        if "1" in rep:
            prec = rep["1"]["precision"]
            rec = rep["1"]["recall"]
            f1 = rep["1"]["f1-score"]
        else:
            prec = rec = f1 = np.nan

        rows.append({
            "Algorithm": name,
            "Train Accuracy": res["train_acc"],
            "Test Accuracy": res["test_acc"],
            "ROC AUC": res["roc_auc"],
            "Precision (class 1)": prec,
            "Recall (class 1)": rec,
            "F1-score (class 1)": f1
        })

    comp_df = pd.DataFrame(rows)
    return comp_df

# --------- Model export / import helpers ---------

def export_model(pipeline, filename="best_model.joblib"):
    """Saves the trained model pipeline to a file."""
    try:
        joblib.dump(pipeline, filename)
        return f"Model successfully exported to {filename}"
    except Exception as e:
        return f"Error exporting model: {e}"

def get_model_bytes(pipeline):
    """Serializes the model pipeline to an in-memory byte buffer."""
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)  # Rewind the buffer to the beginning
    return buffer.read()

def load_uploaded_pipeline(file_obj):
    """
    Load a pipeline object from a generic file object (uploaded by Streamlit).
    """
    return joblib.load(file_obj)

# ---------------------------------------------------------
# NEW: Production Prediction Engine
# ---------------------------------------------------------

def predict_length_of_stay(model_pipeline, input_data: dict):
    """
    Takes a dictionary of 23 raw user inputs, preprocesses them 
    (converting text to numbers), and returns the model's prediction.
    """
    
    # 1. Define the exact feature list in the order the model expects.
    feature_order = [
        'rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 
        'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother', 
        'fibrosisandother', 'malnutrition', 'hemo', 'hematocrit', 'neutrophils', 
        'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 
        'respiration', 'secondarydiagnosisnonicd9'
    ]
    
    # 2. Preprocess the Raw Inputs (Mapping Logic)
    processed_data = input_data.copy()
    
    # Gender: Male -> 1, Female -> 0
    if isinstance(processed_data.get('gender'), str):
        processed_data['gender'] = 1 if processed_data['gender'].lower().startswith('m') else 0
        
    # RCount: Handle "5+"
    if str(processed_data.get('rcount')) == "5+":
        processed_data['rcount'] = 5
    else:
        processed_data['rcount'] = int(processed_data['rcount'])

    # Ensure all Yes/No flags are integers (0 or 1)
    binary_cols = [
        'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence',
        'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 
        'malnutrition', 'hemo'
    ]
    for col in binary_cols:
        val = processed_data.get(col)
        if isinstance(val, str):
            processed_data[col] = 1 if val.lower() in ['yes', 'true', '1'] else 0
        else:
            processed_data[col] = int(val)

    # 3. Create DataFrame for Prediction
    try:
        df_single = pd.DataFrame([processed_data])
        df_single = df_single[feature_order]
    except KeyError as e:
        return {"error": f"Missing required data field: {e}"}

    # 4. Make Prediction
    try:
        prediction_class = model_pipeline.predict(df_single)[0]
        
        # Try to get probability if the model supports it
        if hasattr(model_pipeline, "predict_proba"):
            probability = model_pipeline.predict_proba(df_single)[0][1] # Prob of class 1 (Long Stay)
        else:
            probability = float(prediction_class)

        result_label = "Long Stay (>7 days)" if prediction_class == 1 else "Short Stay (0-7 days)"
        
        return {
            "prediction_label": result_label,
            "prediction_class": int(prediction_class),
            "probability_long_stay": round(probability, 4)
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}