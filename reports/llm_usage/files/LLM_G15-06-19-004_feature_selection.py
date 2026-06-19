import argparse
import json
import time
import pandas as pd
import numpy as np
import dimod
import neal

def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    allowance: int = 1,
    seed: int = 42,
    percSelected: float = 0.20,
    alpha_computations: int = 100
):
    """
    Selects features using Quantum Unconstrained Binary Optimization (QUBO) based on 
    Spearman correlation. Splits the resulting dataset into training and test sets.
    """
    
    # ---------------------------------------------------------
    # 1. Load Dataset & Calculate Correlations
    # ---------------------------------------------------------
    df = pd.read_csv(normalized_csv)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    start_q_time = time.time()
    
    # Calculate absolute Spearman correlation matrix
    # Note: df.corr(method='spearman') relies on scipy under the hood
    corr_matrix = df.corr(method='spearman').abs()
    
    # Fill any NaNs (e.g., from zero-variance features) with 0
    corr_matrix = corr_matrix.fillna(0)
    
    # rho_V: Feature relevance to the target
    rho_v = corr_matrix[target_column].drop(index=target_column)
    
    # rho_F: Feature redundancy (feature-to-feature correlation)
    rho_f = corr_matrix.drop(index=target_column, columns=target_column)
    
    q_matrix_creation_time = time.time() - start_q_time
    
    features = list(rho_v.index)
    n_features = len(features)
    
    # Determine the target number of features (K)
    target_k = int(round(percSelected * n_features))
    target_k = max(1, target_k) # Ensure we select at least 1 feature

    # ---------------------------------------------------------
    # 2. QUBO Setup & Alpha Binary Search
    # ---------------------------------------------------------
    sampler = neal.SimulatedAnnealingSampler()
    
    low, high = 0.0, 1.0
    logs = []
    optim_times = []
    
    closest_diff = float('inf')
    best_state = None
    
    for iteration in range(alpha_computations):
        alpha = (low + high) / 2.0
        
        # Build the QUBO formulation: minimize f(x) = -x^T Q x
        # Diagonal Q elements (Relevance): alpha * |rho_v|
        # Off-diagonal Q elements (Redundancy): -(1 - alpha) * |rho_f|
        # Since we minimize -x^T Q x:
        # Linear terms h_i = -alpha * |rho_v,i|
        # Quadratic terms J_ij = 2 * (1 - alpha) * |rho_f,ij| (multiplying by 2 for matrix symmetry)
        
        h = {f: -alpha * rho_v[f] for f in features}
        J = {}
        
        # Extract upper triangle for quadratic terms to avoid double counting
        for idx1, f1 in enumerate(features):
            for idx2, f2 in enumerate(features):
                if idx1 < idx2:
                    J[(f1, f2)] = 2.0 * (1.0 - alpha) * rho_f.loc[f1, f2]
        
        # Create Binary Quadratic Model
        bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY)
        
        # Sample the problem
        start_opt_time = time.time()
        sampleset = sampler.sample(bqm, seed=seed)
        opt_time = time.time() - start_opt_time
        
        optim_times.append(opt_time)
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy
        
        # Parse selected features
        selected_features = [f for f, val in best_sample.items() if val == 1]
        n_selected = len(selected_features)
        
        # Log the current iteration
        logs.append({
            "alpha": alpha,
            "optimization_time": opt_time,
            "n_selected": n_selected,
            "cost_value": energy
        })
        
        # Track the best configuration closest to target_k
        diff = abs(n_selected - target_k)
        if diff < closest_diff:
            closest_diff = diff
            best_state = (alpha, selected_features, n_selected, iteration + 1)
        
        # Binary Search Logic
        if diff <= allowance:
            break # We found a valid configuration within allowance
        elif n_selected < target_k:
            # Too few features selected -> Increase alpha to prioritize relevance more
            low = alpha 
        else:
            # Too many features selected -> Decrease alpha to penalize redundancy more
            high = alpha
            
    # Unpack the best state found during the search
    best_alpha, best_selected, best_n_selected, iterations_used = best_state

    # ---------------------------------------------------------
    # 3. Log Optimizations to CSV
    # ---------------------------------------------------------
    # Sort logs by ascending alpha value
    logs_df = pd.DataFrame(logs).sort_values(by="alpha")
    logs_df.to_csv(output_ottim_csv, index=False)

    # ---------------------------------------------------------
    # 4. Dataset Reduction & Splitting
    # ---------------------------------------------------------
    # Keep only selected features + the target column
    columns_to_keep = best_selected + [target_column]
    reduced_df = df[columns_to_keep]
    
    total_samples = len(reduced_df)
    test_size = int(total_samples * percTest)
    train_size = total_samples - test_size
    
    # CRITICAL: Clean cut split, NO shuffling
    train_df = reduced_df.iloc[:train_size]
    test_df = reduced_df.iloc[train_size:]
    
    # Save the splits
    train_df.to_csv(reducedTrain_csv, index=False)
    test_df.to_csv(reducedTest_csv, index=False)

    # ---------------------------------------------------------
    # 5. Output JSON Statistics
    # ---------------------------------------------------------
    # Generate binary selected vector matching the original feature order
    selected_vector = [1 if f in best_selected else 0 for f in features]
    
    stats = {
        "n_features": n_features,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": best_n_selected,
        "alpha": round(best_alpha, 4),
        "selected_vector": selected_vector,
        "selected_feature_names": best_selected,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": iterations_used,
        "percTest": percTest,
        "training_dataset_size": train_size,
        "test_dataset_size": test_size,
        "q_matrix_creation_time": round(q_matrix_creation_time, 4),
        "mean_optimization_time": round(float(np.mean(optim_times)), 4),
        "std_dev_optimization_time": round(float(np.std(optim_times)), 4)
    }

    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select features using QUBO and simulated annealing."
    )
    
    parser.add_argument("--in-normalized", type=str, required=True, help="Path to input normalized CSV.")
    parser.add_argument("--out-train", type=str, required=True, help="Path to save reduced training CSV.")
    parser.add_argument("--out-test", type=str, required=True, help="Path to save reduced test CSV.")
    parser.add_argument("--out-optimizations", type=str, required=True, help="Path to save alpha iterations log.")
    parser.add_argument("--out-json", type=str, required=True, help="Path to save execution statistics JSON.")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--perc-selected", type=float, default=0.20, help="Target proportion of features to select.")
    parser.add_argument("--allowance", type=int, default=1, help="Tolerance for the number of selected features.")
    parser.add_argument("--perc-test", type=float, default=0.30, help="Proportion of the dataset to reserve for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the Simulated Annealing sampler.")
    parser.add_argument("--alpha-computations", type=int, default=100, help="Maximum number of alpha binary search iterations.")
    
    args = parser.parse_args()
    
    select_features(
        normalized_csv=args.in_normalized,
        reducedTrain_csv=args.out_train,
        reducedTest_csv=args.out_test,
        output_ottim_csv=args.out_optimizations,
        output_json=args.out_json,
        target_column=args.target,
        percTest=args.perc_test,
        allowance=args.allowance,
        seed=args.seed,
        percSelected=args.perc_selected,
        alpha_computations=args.alpha_computations
    )

