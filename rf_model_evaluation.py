"""
gene_ranking_rf.py

Performs Random Forest classification repeatedly with different random seeds 
to evaluate model stability and average performance metrics.

Author: Kumanan
Date: 2025-10-30
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
df = pd.read_csv('Dataset1.csv')
df.set_index('GeneID', inplace=True)


# -------------------------------------------------------------------------
# Function to run Random Forest multiple times with varying random states
# -------------------------------------------------------------------------
def run_random_forest_multiple_times(num_iterations=20):
    """
    Runs a Random Forest classifier multiple times using different random seeds.
    Calculates and stores performance metrics for each iteration.
    
    Args:
        num_iterations (int): Number of runs with different random_state values.
        
    Returns:
        pd.DataFrame: DataFrame containing Accuracy, Precision, Recall, F1 Score, and AUC-ROC for each run.
    """
    metric_results = []

    random_state_values = range(100, 100 + num_iterations)

    for seed in random_state_values:
        X = df.drop('Association', axis=1)
        y = df['Association']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Initialize Random Forest classifier with custom hyperparameters
        rf_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=seed
        )

        # Train the model
        rf_classifier.fit(X_train, y_train)

        # Predictions
        y_pred = rf_classifier.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])

        metric_results.append([accuracy, precision, recall, f1, roc_auc])

    # Compile results
    df_metrics = pd.DataFrame(
        metric_results, 
        columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    )
    return df_metrics


# -------------------------------------------------------------------------
# Run model evaluation
# -------------------------------------------------------------------------
result_df = run_random_forest_multiple_times(num_iterations=20)

# Compute descriptive statistics
avg = result_df.mean()
variance = result_df.var()
std_dev = result_df.std()

# Combine into a summary table
summary_df = pd.concat([
    avg.rename('Average'),
    variance.rename('Variance'),
    std_dev.rename('Standard Deviation')
], axis=1)

# Display the summary
print(summary_df)