import os
import json
import pytest
import joblib
import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the core modules of the ML pipeline
from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features
from src.qubo_project.model import train, predict

def test_full_ml_pipeline(tmp_path):
    """
    End-to-End integration test for the QUBO Classification Pipeline.
    Verifies the 7 mandatory conditions sequentially.
    """
    
    # ---------------------------------------------------------
    # 0. Setup & Path Definitions
    # ---------------------------------------------------------
    input_dataset = os.path.join(os.getcwd(), "data", "sample_test_dataset.csv")
    target_col = "target"
    
    # Ensure the dummy dataset exists before running the test
    assert os.path.exists(input_dataset), f"CRITICAL: Dummy dataset missing at {input_dataset}"

    # Use Pytest's tmp_path to dynamically assign safe paths for all artifacts
    normalized_csv = tmp_path / "normalized.csv"
    prep_json = tmp_path / "preprocessing_res.json"
    
    reduced_train_csv = tmp_path / "reduced_train.csv"
    reduced_test_csv = tmp_path / "reduced_test.csv"
    optim_csv = tmp_path / "optimizations.csv"
    fs_json = tmp_path / "feature_selection_res.json"
    
    model_path = tmp_path / "model.joblib"
    train_json = tmp_path / "training_metrics.json"
    
    predictions_csv = tmp_path / "predictions.csv"
    stats_json = tmp_path / "classification_stats.json"

    # ---------------------------------------------------------
    # Phase 1: Preprocessing (Tests Conditions 1, 2, 3)
    # ---------------------------------------------------------
    fit_normalize(
        input_csv=input_dataset,
        target_column=target_col,
        normalized_csv=str(normalized_csv),
        outInitalRes_json=str(prep_json),
        minPercValid=0.05
    )
    
    assert normalized_csv.exists(), "Normalized CSV was not created."
    df_norm = pd.read_csv(normalized_csv)
    
    with open(prep_json, 'r') as f:
        prep_stats = json.load(f)

    # Condition 1: Preprocessing produces a dataset with strictly numeric columns
    assert all(pd.api.types.is_numeric_dtype(df_norm[col]) for col in df_norm.columns), \
        "Not all columns in the normalized dataset are numeric."

    # Condition 2: Successfully handles NaNs and drops sparse columns
    assert df_norm.isna().sum().sum() == 0, "There are still NaN values present in the normalized dataset."
    assert "dropped_feature_names" in prep_stats, "JSON does not track dropped sparse columns."
    assert isinstance(prep_stats["dropped_feature_names"], list), "Dropped features tracking must be a list."

    # Condition 3: Normalization produces a valid dataset (Z-score standardization properties)
    features_only = df_norm.drop(columns=[target_col])
    # Z-score means should be approximately 0
    assert np.allclose(features_only.mean(), 0, atol=1e-2), "Z-score normalization failed: Means are not 0."
    # Z-score standard deviations should be approximately 1 (using ddof=0 as used by StandardScaler)
    assert np.allclose(features_only.std(ddof=0), 1, atol=1e-2), "Z-score normalization failed: Std Devs are not 1."

    # ---------------------------------------------------------
    # Phase 2: QUBO Feature Selection (Tests Conditions 4, 5)
    # ---------------------------------------------------------
    allowance_val = 1
    perc_selected_val = 0.20
    
    select_features(
        normalized_csv=str(normalized_csv),
        reducedTrain_csv=str(reduced_train_csv),
        reducedTest_csv=str(reduced_test_csv),
        output_ottim_csv=str(optim_csv),
        output_json=str(fs_json),
        target_column=target_col,
        percTest=0.30,
        allowance=allowance_val,
        seed=42,
        percSelected=perc_selected_val,
        alpha_computations=10  # Kept low to ensure the test runs quickly
    )

    assert reduced_train_csv.exists() and reduced_test_csv.exists(), "Data splits were not created."
    
    with open(fs_json, 'r') as f:
        fs_stats = json.load(f)

    # Condition 4: Produces a binary vector representing selected features
    selected_vector = fs_stats.get("selected_vector", [])
    assert all(val in [0, 1] for val in selected_vector), \
        "The selected_vector is not strictly binary (contains values other than 0 or 1)."

    # Condition 5: The number of features selected is approximately 20% of the input features
    n_input_features = fs_stats["n_features"]
    n_selected = fs_stats["n_selected"]
    expected_k = max(1, int(round(perc_selected_val * n_input_features)))
    
    # We test that it falls within the allowed mathematical tolerance defined by the allowance parameter
    assert abs(n_selected - expected_k) <= allowance_val, \
        f"Selected features ({n_selected}) is outside the 20% target ({expected_k}) +/- allowance ({allowance_val})."

    # ---------------------------------------------------------
    # Phase 3: Model Training (Tests Condition 6)
    # ---------------------------------------------------------
    train(
        classifier="random_forest",
        reducedTrain_csv=str(reduced_train_csv),
        target_column=target_col,
        model_path=str(model_path),
        metrics_json=str(train_json),
        seed=42
    )

    # Condition 6: Successfully produces and saves a model file (.joblib)
    assert model_path.exists(), "The training phase did not save the model to disk."
    assert model_path.suffix == ".joblib", "The model was not saved with the correct .joblib extension."
    
    # Verify the model can actually be loaded
    try:
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None, "Loaded model object is None."
    except Exception as e:
        pytest.fail(f"Failed to load the saved .joblib model: {e}")

    # ---------------------------------------------------------
    # Phase 4: Prediction (Tests Condition 7)
    # ---------------------------------------------------------
    predict(
        reduced_Test_csv=str(reduced_test_csv),
        target_column=target_col,
        model_path=str(model_path),
        predictions_csv=str(predictions_csv),
        classif_stats_json=str(stats_json)
    )

    assert predictions_csv.exists(), "Predictions CSV was not generated."
    df_preds = pd.read_csv(predictions_csv)

    # Condition 7: Produces a CSV containing exactly specific columns
    expected_columns = ["row_n", "target", "prediction", "score"]
    assert list(df_preds.columns) == expected_columns, \
        f"Prediction columns mismatch. Expected: {expected_columns}, Got: {list(df_preds.columns)}"