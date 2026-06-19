import argparse
import json
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05
):
    """
    Reads a dataset, removes features with insufficient valid data (non-zero, non-NaN),
    applies z-score standardization to the remaining features, and saves the results.
    
    Args:
        input_csv (str): Path to the input dataset (CSV format).
        target_column (str): Name of the target column to exclude from feature transformations.
        normalized_csv (str): Path to save the processed, normalized dataset.
        outInitalRes_json (str): Path to save the processing statistics in JSON format.
        minPercValid (float): Minimum percentage (0.0 to 1.0) of valid data required to keep a feature.
    """
    
    # ---------------------------------------------------------
    # 1. Dataset Input Phase
    # ---------------------------------------------------------
    start_input_time = time.time()
    
    # Read the dataset (assumes first row contains headers by default)
    df = pd.read_csv(input_csv)
    
    end_input_time = time.time()
    dataset_input_time = end_input_time - start_input_time

    # ---------------------------------------------------------
    # 2. Dataset Processing Phase
    # ---------------------------------------------------------
    start_processing_time = time.time()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Separate target from features
    target_data = df[target_column]
    features_df = df.drop(columns=[target_column])

    n_input_features = len(features_df.columns)
    dataset_size = len(df)

    # Identify valid data: non-missing (notna) AND non-zero
    valid_mask = features_df.notna() & (features_df != 0)
    
    # Calculate the ratio of valid data for each feature
    valid_ratio = valid_mask.sum() / dataset_size

    # Filter features based on the minimum valid percentage threshold
    features_to_keep = valid_ratio[valid_ratio >= minPercValid].index.tolist()
    dropped_feature_names = valid_ratio[valid_ratio < minPercValid].index.tolist()

    # Drop the invalid features
    features_df = features_df[features_to_keep]
    n_kept_features = len(features_to_keep)

    # Apply Z-score standardization (Zero mean, Unit variance)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_df)
    
    # Convert the standardized numpy array back to a DataFrame with original column names
    normalized_features_df = pd.DataFrame(
        normalized_features, 
        columns=features_to_keep, 
        index=features_df.index
    )

    # Reattach the target column to the normalized features
    final_df = pd.concat([normalized_features_df, target_data], axis=1)

    # Save the processed dataset to the specified CSV output path
    final_df.to_csv(normalized_csv, index=False)

    end_processing_time = time.time()
    dataset_processing_time = end_processing_time - start_processing_time

    # ---------------------------------------------------------
    # 3. Save JSON Statistics
    # ---------------------------------------------------------
    # Structure the output dictionary strictly as requested
    stats = {
        "n_input_features": n_input_features,
        "n_kept_features": n_kept_features,
        "dataset_size": dataset_size,
        "dataset_input_time": round(dataset_input_time, 2),
        "dataset_processing_time": round(dataset_processing_time, 2),
        "dropped_feature_names": dropped_feature_names
    }

    # Write the statistics to the specified JSON path
    with open(outInitalRes_json, 'w') as json_file:
        json.dump(stats, json_file, indent=4)


if __name__ == "__main__":
    # Command Line Interface Setup
    parser = argparse.ArgumentParser(
        description="Preprocess machine learning datasets for QUBO feature reduction."
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input CSV dataset."
    )
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        help="Name of the target column in the dataset."
    )
    parser.add_argument(
        "--out-data", 
        type=str, 
        required=True, 
        help="Path where the normalized CSV dataset will be saved."
    )
    parser.add_argument(
        "--out-json", 
        type=str, 
        required=True, 
        help="Path where the preprocessing statistics JSON will be saved."
    )
    parser.add_argument(
        "--min-perc-valid", 
        type=float, 
        default=0.05, 
        help="Minimum percentage of valid data (non-zero and non-NaN) required to retain a feature."
    )

    # Parse arguments from the command line
    args = parser.parse_args()

    # Execute the core function using parsed arguments
    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid
    )