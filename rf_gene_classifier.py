"""
Title: Gene Prioritization using Random Forest-based Probability Ranking
Author: Kumanan N Govaichelvan
Date: 2025-10-30

Description:
    This script performs gene prioritization using Random Forest classification.
    It estimates the probability of gene association across multiple iterations 
    to produce a stable and robust ranking of candidate genes.
    
    The final output is a ranked list of genes based on their average 
    probability of association across iterations.

Output:
    - Genes_List.csv : Ranked list of genes with average probabilities.

Usage:
    python rf_gene_prioritization.py
"""

# ===============================================================
# 1. Import Required Libraries
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ===============================================================
# 2. Load and Preprocess Dataset
# ===============================================================
# Path to input dataset
data_path = "Dataset1.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Ensure GeneID is set as index
df.set_index("GeneID", inplace=True)

# Split into features (X) and target labels (y)
X = df.drop(columns=["Association"])
y = df["Association"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================================
# 3. Initialize Structures for Probability Tracking
# ===============================================================
# DataFrame to accumulate probabilities for each gene
probabilities_df = pd.DataFrame(index=X.index, columns=["Average_Probability"], data=0.0)

# Lists to store iteration results
ranked_genes_all_iterations = []   # For consistency checking
avg_probabilities = []             # For optional visualization
auc_roc_final_iteration = None     # Store AUC-ROC for final iteration

# ===============================================================
# 4. Iterative Random Forest Training and Probability Ranking
# ===============================================================
iterations = 1000
print(f"Starting {iterations} iterations of Random Forest training...\n")

for i in range(iterations):
    print(f"Iteration {i + 1}/{iterations}...")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=i
    )

    # Initialize Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=i,
    )

    # Train model
    clf.fit(X_train, y_train)

    # Predict probabilities for all genes
    probabilities = clf.predict_proba(X_scaled)[:, 1]

    # Accumulate probabilities across iterations
    probabilities_df["Average_Probability"] += probabilities

    # Store ranked genes for this iteration
    iteration_ranked = (
        pd.DataFrame({"GeneID": X.index, "Average_Probability": probabilities})
        .sort_values(by="Average_Probability", ascending=False)
    )
    ranked_genes_all_iterations.append(iteration_ranked["GeneID"].tolist())

    # Track probability mean for visualization
    avg_probabilities.append(probabilities_df["Average_Probability"].copy())

    # Calculate AUC-ROC for the final iteration
    if i == iterations - 1:
        auc_roc_final_iteration = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"AUC-ROC (final iteration): {auc_roc_final_iteration:.4f}\n")

# ===============================================================
# 5. Final Averaging and Gene Ranking
# ===============================================================
# Compute mean probability over all iterations
probabilities_df["Average_Probability"] /= iterations

# Add the true labels
probabilities_df["Association"] = y

# Rank genes by their average probability (descending order)
ranked_genes = probabilities_df.sort_values(by="Average_Probability", ascending=False)

# Save ranked gene list
output_path = "Genes_List.csv"
ranked_genes.to_csv(output_path)

print("Gene ranking complete.")
print(f"Results saved to: {output_path}")

# ===============================================================
# 6. Optional: Visualize Probability Convergence
# ===============================================================
plt.figure(figsize=(8, 4))
plt.plot([p.mean() for p in avg_probabilities])
plt.xlabel("Iteration")
plt.ylabel("Mean Accumulated Probability")
plt.title("Convergence of Average Probability Across Iterations")
plt.tight_layout()
plt.show()
