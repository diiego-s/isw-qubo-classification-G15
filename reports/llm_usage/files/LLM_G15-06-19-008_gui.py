import os
import json
import pandas as pd
import streamlit as st

# Import the core pipeline functions
from preprocessing import fit_normalize
from feature_selection import select_features
from model import train, predict

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(page_title="QUBO ML Pipeline", layout="wide")

# Ensure the local outputs directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define hardcoded relative paths for pipeline artifacts
PATHS = {
    "uploaded_csv": os.path.join(OUTPUT_DIR, "uploaded_input.csv"),
    "normalized_csv": os.path.join(OUTPUT_DIR, "normalized.csv"),
    "prep_json": os.path.join(OUTPUT_DIR, "preprocessing_res.json"),
    "reduced_train_csv": os.path.join(OUTPUT_DIR, "reduced_train.csv"),
    "reduced_test_csv": os.path.join(OUTPUT_DIR, "reduced_test.csv"),
    "optim_csv": os.path.join(OUTPUT_DIR, "optimizations.csv"),
    "fs_json": os.path.join(OUTPUT_DIR, "feature_selection_res.json"),
    "model_path": os.path.join(OUTPUT_DIR, "model.joblib"),
    "train_json": os.path.join(OUTPUT_DIR, "training_metrics.json"),
    "predictions_csv": os.path.join(OUTPUT_DIR, "predictions.csv"),
    "stats_json": os.path.join(OUTPUT_DIR, "classification_stats.json")
}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# ---------------------------------------------------------
# UI Layout: Sidebar for Parameters
# ---------------------------------------------------------
st.sidebar.title("Pipeline Parameters")
target_col = st.sidebar.text_input("Target Column Name", value="target")

st.sidebar.subheader("1. Preprocessing")
min_perc_valid = st.sidebar.slider("Minimum Valid Percentage (minPercValid)", 0.0, 1.0, 0.05, 0.01)

st.sidebar.subheader("2. QUBO Feature Selection")
perc_selected = st.sidebar.slider("Percentage of Features to Select", 0.01, 1.0, 0.20, 0.01)
perc_test = st.sidebar.slider("Test Set Percentage", 0.1, 0.5, 0.30, 0.05)
allowance = st.sidebar.number_input("Target Allowance", min_value=0, max_value=10, value=1)
alpha_comps = st.sidebar.number_input("Alpha Computations", min_value=10, max_value=500, value=100)
qubo_seed = st.sidebar.number_input("QUBO Seed", value=42)

st.sidebar.subheader("3. Model Training")
classifier = st.sidebar.selectbox("Classifier", ["random_forest", "logistic_regression", "gradient_boosting"])
model_seed = st.sidebar.number_input("Model Seed", value=42)

# ---------------------------------------------------------
# Main UI Layout: Sequential Execution Steps
# ---------------------------------------------------------
st.title("QUBO Binary Classification Pipeline")
st.markdown("Run your machine learning pipeline sequentially below.")

# --- STEP 1: DATA UPLOAD ---
st.header("Step 1: Dataset Upload")
uploaded_file = st.file_uploader("Upload your raw dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv(PATHS["uploaded_csv"], index=False)
    st.success(f"Dataset successfully uploaded and saved to `{PATHS['uploaded_csv']}`.")
    with st.expander("Preview Dataset"):
        st.dataframe(df.head())

st.divider()

# --- STEP 2: PREPROCESSING ---
st.header("Step 2: Preprocessing")
if st.button("Run Preprocessing"):
    if not os.path.exists(PATHS["uploaded_csv"]):
        st.error("Please upload a dataset in Step 1 first.")
    elif target_col not in pd.read_csv(PATHS["uploaded_csv"], nrows=1).columns:
        st.error(f"Target column '{target_col}' not found in the uploaded dataset.")
    else:
        with st.spinner("Running Preprocessing..."):
            fit_normalize(
                input_csv=PATHS["uploaded_csv"],
                target_column=target_col,
                normalized_csv=PATHS["normalized_csv"],
                outInitalRes_json=PATHS["prep_json"],
                minPercValid=min_perc_valid
            )
        st.success("Preprocessing Complete!")

if os.path.exists(PATHS["prep_json"]):
    with st.expander("View Preprocessing Results"):
        prep_stats = load_json(PATHS["prep_json"])
        st.json(prep_stats)

st.divider()

# --- STEP 3: QUBO FEATURE SELECTION ---
st.header("Step 3: QUBO Feature Selection")
if st.button("Run Feature Selection"):
    if not os.path.exists(PATHS["normalized_csv"]):
        st.error("Please run Preprocessing first to generate the normalized dataset.")
    else:
        with st.spinner("Running QUBO Feature Selection... (This may take a while)"):
            select_features(
                normalized_csv=PATHS["normalized_csv"],
                reducedTrain_csv=PATHS["reduced_train_csv"],
                reducedTest_csv=PATHS["reduced_test_csv"],
                output_ottim_csv=PATHS["optim_csv"],
                output_json=PATHS["fs_json"],
                target_column=target_col,
                percTest=perc_test,
                allowance=allowance,
                seed=qubo_seed,
                percSelected=perc_selected,
                alpha_computations=alpha_comps
            )
        st.success("Feature Selection Complete!")

if os.path.exists(PATHS["fs_json"]):
    with st.expander("View Feature Selection Results"):
        fs_stats = load_json(PATHS["fs_json"])
        st.metric(label="Features Selected", value=fs_stats.get("n_selected", 0))
        st.json(fs_stats)

st.divider()

# --- STEP 4: MODEL TRAINING ---
st.header("Step 4: Model Training")
if st.button("Run Model Training"):
    if not os.path.exists(PATHS["reduced_train_csv"]):
        st.error("Please run Feature Selection first to generate the training dataset.")
    else:
        with st.spinner(f"Training {classifier}..."):
            train(
                classifier=classifier,
                reducedTrain_csv=PATHS["reduced_train_csv"],
                target_column=target_col,
                model_path=PATHS["model_path"],
                metrics_json=PATHS["train_json"],
                seed=model_seed
            )
        st.success("Model Training Complete!")

if os.path.exists(PATHS["train_json"]):
    with st.expander("View Training Metrics"):
        train_stats = load_json(PATHS["train_json"])
        st.json(train_stats)

st.divider()

# --- STEP 5: PREDICTIONS & EVALUATION ---
st.header("Step 5: Predictions & Evaluation")
if st.button("Run Predictions"):
    if not os.path.exists(PATHS["model_path"]) or not os.path.exists(PATHS["reduced_test_csv"]):
        st.error("Ensure both the trained model and test dataset exist (Run steps 3 and 4).")
    else:
        with st.spinner("Evaluating Test Set..."):
            predict(
                reduced_Test_csv=PATHS["reduced_test_csv"],
                target_column=target_col,
                model_path=PATHS["model_path"],
                predictions_csv=PATHS["predictions_csv"],
                classif_stats_json=PATHS["stats_json"]
            )
        st.success("Predictions Complete!")

if os.path.exists(PATHS["stats_json"]):
    with st.expander("View Classification Stats & Confusion Matrix"):
        class_stats = load_json(PATHS["stats_json"])
        
        # Display Core Metrics
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{class_stats.get('accuracy', 0) * 100:.1f}%")
        col2.metric("ROC AUC", f"{class_stats.get('roc_auc', 0) * 100:.1f}%")
        
        # Display Confusion Matrix cleanly
        st.subheader("Confusion Matrix")
        cm_data = class_stats.get("confusion_matrix", {}).get("matrix", [])
        if cm_data:
            cm_df = pd.DataFrame(
                cm_data, 
                columns=["Predicted Class 0", "Predicted Class 1"],
                index=["Actual Class 0", "Actual Class 1"]
            )
            st.table(cm_df)
        
        st.subheader("Full JSON Output")
        st.json(class_stats)