import os
import json
import pandas as pd
import streamlit as st

# Import the core pipeline functions
from preprocessing import fit_normalize
from feature_selection import select_features
from model import train, predict

# ---------------------------------------------------------
# Configuration & State Initialization
# ---------------------------------------------------------
st.set_page_config(page_title="QUBO Machine Learning Pipeline", page_icon="🧪", layout="wide")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define static paths
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

# Initialize wizard state flags
for step in ["step_1_done", "step_2_done", "step_3_done", "step_4_done", "step_5_done"]:
    if step not in st.session_state:
        st.session_state[step] = False

# Helper function
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

# ---------------------------------------------------------
# Sidebar: Global Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Global Settings")
    target_col = st.text_input("Target Column Name", value="target", help="The dependent variable to predict.")
    st.info("Advanced configurations for each phase are available in expandable panels within the main workflow.")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.title("QUBO Feature Selection Pipeline")
st.markdown("Follow the sequential steps below to process data, optimize feature selection using Quantum algorithms, and evaluate models.")
st.divider()

# ---------------------------------------------------------
# STEP 1: Dataset Upload
# ---------------------------------------------------------
st.header("Step 1: Data Initialization")
uploaded_file = st.file_uploader("Upload your raw dataset (.csv)", type=["csv"], help="Ensure the file is comma-separated and contains headers.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if target_col not in df.columns:
            st.error(f"Validation Error: Target column '{target_col}' not found in the dataset.")
            st.session_state.step_1_done = False
        else:
            df.to_csv(PATHS["uploaded_csv"], index=False)
            st.session_state.step_1_done = True
            
            total_rows, total_cols = df.shape
            st.success(f"Dataset successfully loaded: {total_rows:,} rows | {total_cols} columns")
            with st.expander("Preview Initial Data"):
                # Clarify that the table below is only a partial view
                st.caption("Showing a preview of the first 5 rows:")
                st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.session_state.step_1_done = False
else:
    st.session_state.step_1_done = False

st.divider()

# ---------------------------------------------------------
# STEP 2: Preprocessing
# ---------------------------------------------------------
if st.session_state.step_1_done:
    st.header("Step 2: Preprocessing & Cleaning")
    
    with st.expander("⚙️ Advanced Settings: Preprocessing"):
        min_perc_valid = st.slider("Minimum Valid Data Threshold", 0.0, 1.0, 0.05, 0.01, help="Drops features with excessive zeros or missing values.")
        
    if st.button("Run Preprocessing", type="primary"):
        with st.spinner("Standardizing features and filtering sparsity..."):
            try:
                fit_normalize(
                    input_csv=PATHS["uploaded_csv"],
                    target_column=target_col,
                    normalized_csv=PATHS["normalized_csv"],
                    outInitalRes_json=PATHS["prep_json"],
                    minPercValid=min_perc_valid
                )
                st.session_state.step_2_done = True
                st.success("Preprocessing completed.")
            except Exception as e:
                st.error(f"Preprocessing failed: {str(e)}")

    if st.session_state.step_2_done:
        stats = load_json(PATHS["prep_json"])
        if stats:
             cols = st.columns(3)
             cols[0].metric("Initial Features", stats.get("n_input_features", 0))
             cols[1].metric("Retained Features", stats.get("n_kept_features", 0))
             cols[2].metric("Dropped Features", len(stats.get("dropped_feature_names", [])))
else:
    st.info("Awaiting Dataset Upload to unlock Step 2.")
    
st.divider()

# ---------------------------------------------------------
# STEP 3: QUBO Feature Selection
# ---------------------------------------------------------
if st.session_state.step_2_done:
    st.header("Step 3: QUBO Feature Selection")
    
    with st.expander("⚙️ Advanced Settings: QUBO Optimization"):
        col1, col2 = st.columns(2)
        with col1:
            perc_selected = st.slider("Target Feature %", 0.01, 1.0, 0.20, 0.01)
            perc_test = st.slider("Test Set %", 0.1, 0.5, 0.30, 0.05)
        with col2:
            allowance = st.number_input("Target Allowance", min_value=0, max_value=10, value=1)
            alpha_comps = st.number_input("Alpha Iterations", min_value=10, max_value=500, value=100)
            qubo_seed = st.number_input("QUBO Seed", value=42)

    if st.button("Run Quantum Feature Selection", type="primary"):
        with st.spinner("Calculating correlation matrices and executing simulated annealing..."):
            try:
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
                st.session_state.step_3_done = True
                st.success("Feature selection optimization successful.")
            except Exception as e:
                 st.error(f"Optimization failed: {str(e)}")

    if st.session_state.step_3_done:
        fs_stats = load_json(PATHS["fs_json"])
        if fs_stats:
            cols = st.columns(4)
            cols[0].metric("Target K", fs_stats.get("target_k", 0))
            cols[1].metric("Selected K", fs_stats.get("n_selected", 0))
            cols[2].metric("Best Alpha", fs_stats.get("alpha", 0.0))
            cols[3].metric("Train Size", fs_stats.get("training_dataset_size", 0))
else:
    st.info("Awaiting Preprocessing to unlock Step 3.")

st.divider()

# ---------------------------------------------------------
# STEP 4: Model Training
# ---------------------------------------------------------
if st.session_state.step_3_done:
    st.header("Step 4: Model Training")
    
    classifier = st.selectbox("Select Classification Algorithm", ["random_forest", "logistic_regression", "gradient_boosting"])
    
    with st.expander("⚙️ Advanced Settings: Training"):
        model_seed = st.number_input("Model Seed", value=42, key="train_seed")

    if st.button("Train Model", type="primary"):
        with st.spinner(f"Fitting {classifier.replace('_', ' ').title()}..."):
            try:
                train(
                    classifier=classifier,
                    reducedTrain_csv=PATHS["reduced_train_csv"],
                    target_column=target_col,
                    model_path=PATHS["model_path"],
                    metrics_json=PATHS["train_json"],
                    seed=model_seed
                )
                st.session_state.step_4_done = True
                st.success("Model successfully trained and serialized.")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
else:
    st.info("Awaiting Feature Selection to unlock Step 4.")

st.divider()

# ---------------------------------------------------------
# STEP 5 & 6: Prediction & Dashboard
# ---------------------------------------------------------
if st.session_state.step_4_done:
    st.header("Step 5 & 6: Prediction & Final Dashboard")
    
    if st.button("Generate Predictions & Evaluate", type="primary"):
        with st.spinner("Scoring test dataset..."):
            try:
                predict(
                    reduced_Test_csv=PATHS["reduced_test_csv"],
                    target_column=target_col,
                    model_path=PATHS["model_path"],
                    predictions_csv=PATHS["predictions_csv"],
                    classif_stats_json=PATHS["stats_json"]
                )
                st.session_state.step_5_done = True
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

    if st.session_state.step_5_done:
        stats = load_json(PATHS["stats_json"])
        if stats:
            st.markdown("### 📊 Performance Metrics")
            
            # Top-level airy metric dashboard
            col1, col2, col3, col4 = st.columns(4)
            acc = stats.get('accuracy', 0)
            auc = stats.get('roc_auc', 0)
            col1.metric("Overall Accuracy", f"{acc * 100:.1f}%")
            col2.metric("ROC AUC", f"{auc * 100:.1f}%")
            col3.metric("Test Samples", stats.get("n_samples", 0))
            col4.metric("Positives Found", stats.get("target_1_count", 0))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed row for classes
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Class 0 (Negative) Metrics")
                c_stats = stats.get("class_0", {})
                st.write(f"**Precision:** {c_stats.get('precision', 0):.2f}")
                st.write(f"**Recall:** {c_stats.get('recall', 0):.2f}")
                st.write(f"**F1-Score:** {c_stats.get('f1', 0):.2f}")
                
            with c2:
                st.subheader("Class 1 (Positive) Metrics")
                c_stats = stats.get("class_1", {})
                st.write(f"**Precision:** {c_stats.get('precision', 0):.2f}")
                st.write(f"**Recall:** {c_stats.get('recall', 0):.2f}")
                st.write(f"**F1-Score:** {c_stats.get('f1', 0):.2f}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Clean Confusion Matrix Display
            st.subheader("Confusion Matrix")
            cm_data = stats.get("confusion_matrix", {}).get("matrix", [])
            if cm_data:
                cm_df = pd.DataFrame(
                    cm_data, 
                    columns=["Predicted 0", "Predicted 1"],
                    index=["Actual 0", "Actual 1"]
                )
                st.table(cm_df.style.background_gradient(cmap='Blues', axis=None))
                
            with st.expander("View Raw Output JSONs"):
                st.json(stats)
else:
    st.info("Awaiting Model Training to unlock final evaluation.")