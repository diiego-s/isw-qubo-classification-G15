# Binary Classification Pipeline with QUBO Feature Selection

## Project Description

This project implements an end-to-end Machine Learning pipeline designed for binary classification tasks, enhanced by a **Quantum Unconstrained Binary Optimization (QUBO)** approach to feature selection.

The pipeline addresses high-dimensional data challenges by:

1. **Preprocessing**: Handling sparse data and performing Z-score normalization.
2. **QUBO Feature Selection**: Utilizing Spearman correlation matrices and Simulated Annealing to identify the most relevant features while minimizing redundancy.
3. **Model Training & Prediction**: Providing a robust framework for training classifiers (Random Forest, Logistic Regression, Gradient Boosting) and evaluating performance.
4. **Interactive GUI**: A Streamlit-based interface for managing the pipeline workflow.

---

## Repository Structure

```text
.
├── data/               # Source datasets (e.g., sample_test_dataset.csv)
├── files/              # Supplemental scripts and generated models
├── llm_usage/          # Interaction logs with LLM assistants
├── outputs/            # Pipeline artifacts (CSVs, JSON metrics, model files)
├── reports/            # Analysis and project documentation
├── src/
│   └── qubo_project/   # Core source code
│       ├── gui.py
│       ├── model.py
│       ├── preprocessing.py
│       └── feature_selection.py
├── tests/              # Pytest integration suite
├── README.md
├── group_info.yaml
└── requirements.txt

```

---

## Setup and Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd isw-qubo-classification-G15

```


2. **Create a virtual environment**:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```



---

## How to Run the GUI

For a visual, sequential workflow, use the Streamlit interface:

```bash
streamlit run src/qubo_project/gui.py

```

---

## Command Line Interface (CLI) Usage

You can execute the pipeline components independently via the terminal:

### 1. Preprocessing

```bash
python src/qubo_project/preprocessing.py \
    --input data/sample_test_dataset.csv \
    --target target \
    --out-data outputs/normalized.csv \
    --out-json outputs/preprocessing_res.json \
    --min-perc-valid 0.05

```

### 2. Feature Selection

```bash
python src/qubo_project/feature_selection.py \
    --in-normalized outputs/normalized.csv \
    --out-train outputs/reduced_train.csv \
    --out-test outputs/reduced_test.csv \
    --out-optimizations outputs/optimizations.csv \
    --out-json outputs/feature_selection_res.json \
    --target target

```

### 3. Model Training

```bash
python src/qubo_project/model.py train \
    --classifier random_forest \
    --in-reduced outputs/reduced_train.csv \
    --target target \
    --out-model outputs/model.joblib \
    --out-metrics outputs/training_metrics.json

```

### 4. Prediction

```bash
python src/qubo_project/model.py predict \
    --input-testset outputs/reduced_test.csv \
    --target target \
    --model outputs/model.joblib \
    --out-predictions outputs/predictions.csv \
    --out-stats outputs/classification_stats.json

```

---

## Running Tests

To verify the integrity of the pipeline, run the automatic test suite:

```bash
pytest src/tests/test_pipeline.py -v

```

---

## Authors

**Group G15**

* Diego Serra
* Michele Chillotti