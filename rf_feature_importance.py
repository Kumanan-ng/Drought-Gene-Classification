"""
rf_feature_importance_rf.py

Evaluates Random Forest feature importances across multiple iterations 
to assess feature stability for gene prioritization.

Author: Kumanan
Date: 2025-10-30
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
df = pd.read_csv('DS6_RS.csv')
df.set_index('GeneID', inplace=True)


# -------------------------------------------------------------------------
# Function: run_random_forest_multiple_times
# -------------------------------------------------------------------------
def run_random_forest_multiple_times(X, y, num_iterations=20):
    """
    Runs Random Forest multiple times with different random seeds and 
    collects feature importances from each run.

    Args:
        X (pd.DataFrame): Feature matrix (genes × features)
        y (pd.Series): Target variable (association label)
        num_iterations (int): Number of model runs

    Returns:
        pd.DataFrame: Feature importances for each iteration
    """
    feature_importance_df = pd.DataFrame(index=X.columns)
    random_state_values = range(100, 100 + num_iterations)

    for seed in random_state_values:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Initialize and train Random Forest
        rf_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=seed
        )
        rf_classifier.fit(X_train, y_train)

        # Record feature importances
        feature_importance_df[f"Iteration_{seed}"] = rf_classifier.feature_importances_

    return feature_importance_df


# -------------------------------------------------------------------------
# Prepare data and run the model
# -------------------------------------------------------------------------
X = df.drop('Association', axis=1)
y = df['Association']

feature_importance_df = run_random_forest_multiple_times(X, y, num_iterations=20)

# -------------------------------------------------------------------------
# Save and display results
# -------------------------------------------------------------------------
feature_importance_df.to_csv('rf_feature_importances.csv', index=True)
print("Feature importance results saved to 'rf_feature_importances.csv'.")
print(feature_importance_df.head())
