import argparse
import json
import time
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Training Function
# ---------------------------------------------------------
def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42
):
    """
    Trains a binary classification model on the provided dataset and saves the 
    model along with its execution statistics.
    """
    start_input_time = time.time()
    
    # Load the training dataset
    df_train = pd.read_csv(reducedTrain_csv)
    
    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
    # Separate features and target
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    
    dataset_input_time = time.time() - start_input_time
    
    # Initialize the selected classifier
    clf_name_lower = classifier.lower()
    if clf_name_lower == "random_forest":
        model = RandomForestClassifier(random_state=seed)
    elif clf_name_lower == "logistic_regression":
        model = LogisticRegression(random_state=seed, max_iter=1000)
    elif clf_name_lower == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError("Unsupported classifier. Choose from: 'random_forest', 'logistic_regression', 'gradient_boosting'.")

    # Train the model
    start_train_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_train_time
    
    # Save the trained model to disk
    joblib.dump(model, model_path)
    
    # Calculate dataset statistics
    n_samples = len(y_train)
    n_features = X_train.shape[1]
    target_1_count = int(y_train.sum())
    target_1_percentage = round((target_1_count / n_samples) * 100, 2)
    
    # Structure and save the JSON metrics
    stats = {
        "classifier": clf_name_lower,
        "seed": seed,
        "training_dataset": reducedTrain_csv.split('/')[-1], # extracts filename
        "target_column": target_column,
        "model_path": model_path.split('/')[-1],
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": target_1_percentage,
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }
    
    with open(metrics_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Training complete. Model saved to {model_path}")


# ---------------------------------------------------------
# 2. Prediction Function
# ---------------------------------------------------------
def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str
):
    """
    Loads a trained model, generates predictions on a test set, and calculates
    binary classification metrics.
    """
    # Load dataset
    df_test = pd.read_csv(reduced_Test_csv)
    
    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    # Load the trained model
    model = joblib.load(model_path)
    
    # Determine classifier name for the JSON output based on the loaded object type
    if isinstance(model, RandomForestClassifier):
        clf_name = "random_forest"
    elif isinstance(model, LogisticRegression):
        clf_name = "logistic_regression"
    elif isinstance(model, GradientBoostingClassifier):
        clf_name = "gradient_boosting"
    else:
        clf_name = "unknown_classifier"

    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    
    # Ensure probabilities are extracted properly (specifically for the positive class '1')
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if a model does not support predict_proba natively
        y_prob = y_pred 
        
    # Create the predictions CSV mapping
    predictions_df = pd.DataFrame({
        "row_n": range(len(y_test)),
        "target": y_test.values,
        "prediction": y_pred,
        "score": round(pd.Series(y_prob), 4) # Round probabilities for cleaner output
    })
    
    # Save predictions to CSV
    predictions_df.to_csv(predictions_csv, index=False)
    
    # Calculate classification metrics
    n_samples = len(y_test)
    target_1_count = int(y_test.sum())
    target_1_percentage = round((target_1_count / n_samples) * 100, 2)
    
    acc = round(accuracy_score(y_test, y_pred), 2)
    roc_auc = round(roc_auc_score(y_test, y_prob), 2)
    
    # Get detailed classification report (force zero_division to avoid warnings on sparse predictions)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Safe retrieval of class 0 and 1 stats (keys are strings in output_dict)
    class_0_stats = report.get("0", report.get("0.0", {}))
    class_1_stats = report.get("1", report.get("1.0", {}))
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Structure the JSON stats strictly to the provided requirement
    classif_stats = {
        "classifier": clf_name,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": target_1_percentage,
        "accuracy": acc,
        "class_0": {
            "precision": round(class_0_stats.get("precision", 0.0), 2),
            "recall": round(class_0_stats.get("recall", 0.0), 2),
            "f1": round(class_0_stats.get("f1-score", 0.0), 2),
            "support": int(class_0_stats.get("support", 0))
        },
        "class_1": {
            "precision": round(class_1_stats.get("precision", 0.0), 2),
            "recall": round(class_1_stats.get("recall", 0.0), 2),
            "f1": round(class_1_stats.get("f1-score", 0.0), 2),
            "support": int(class_1_stats.get("support", 0))
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }
    
    with open(classif_stats_json, 'w') as f:
        json.dump(classif_stats, f, indent=4)