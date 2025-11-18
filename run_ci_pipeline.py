import joblib
import json
import pandas as pd
import sys
import time
import argparse

# --- Helper Functions ---
# These functions organize the code cleanly.

def load_prediction_assets():
    """
    Loads the trained model and the feature list from disk.
    These files were created by `train_model.py`.
    """
    print("Loading AI model and feature list...")
    try:
        # Load the model object
        model = joblib.load('ci_model.pkl')
    except FileNotFoundError:
        print("Error: 'ci_model.pkl' not found.")
        print("Please run 'python train_model.py' first.")
        sys.exit(1) # Exit with an error code
        
    try:
        # Load the list of feature names
        with open('model_features.json', 'r') as f:
            features_list = json.load(f)
    except FileNotFoundError:
        print("Error: 'model_features.json' not found.")
        print("Please run 'python train_model.py' first.")
        sys.exit(1) # Exit with an error code
        
    return model, features_list

def prepare_input_data(commit_data, features_list):
    """
    Transforms the single-commit input data (a dict) into the exact
    DataFrame format the model was trained on.
    
    This is the most critical step in a real-world MLOps pipeline!
    This function prevents "feature mismatch" errors.
    """
    # 1. Create a single-row DataFrame from the raw input dictionary
    df = pd.DataFrame(commit_data, index=[0])
    
    # 2. Perform the *exact same* one-hot encoding as in training
    df_processed = pd.get_dummies(df)
    
    # Re-index the DataFrame to match the training columns
    # - It adds any columns the model expects but this commit didn't have
    #   (e.g., 'author_contractor') and sets their value to 0.
    # - It removes any new columns this commit has that the model
    #   never saw in training (e.g., a new author 'dev_x').
    df_final = df_processed.reindex(columns=features_list, fill_value=0)
    
    return df_final

def run_long_test_suite():
    """Simulates a time-consuming CI step (e.g., integration tests)."""
    print("Build predicted to SUCCEED. Running full test suite...")
    print("Running tests (this will take 10 seconds)...")
    for i in range(10):
        time.sleep(1) # Simulate work being done
        print(f"  ... test {i+1}/10 complete ...")
    print("✅ All tests passed!")
    return 0 # POSIX standard for success

def fail_fast():
    """Simulates failing the build early to save time/resources."""
    print("="*50)
    print("🚨 BUILD PREDICTED TO FAIL! 🚨")
    print("Skipping long test suite to save resources.")
    print("Flagging commit for author review.")
    print("="*50)
    return 1 # POSIX standard for failure

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- 1. Parse Input Arguments ---
    parser = argparse.ArgumentParser(description="Simulate a 'smart' CI pipeline run.")
    parser.add_argument('--author', required=True, help="Git commit author.")
    parser.add_argument('--files-changed', required=True, type=int, help="Number of files changed.")
    parser.add_argument('--lines-added', required=True, type=int, help="Lines added.")
    parser.add_argument('--lines-deleted', required=True, type=int, help="Lines deleted.")
    parser.add_argument('--dominant-file-type', required=True, help="e.g., .py, .js, .yml")
    
    args = parser.parse_args()

    # --- 2. Collate Input Data ---
    # Create the raw data dictionary from the parsed arguments
    new_commit_data = {
        'author': args.author,
        'files_changed': args.files_changed,
        'lines_added': args.lines_added,
        'lines_deleted': args.lines_deleted,
        'dominant_file_type': args.dominant_file_type
    }
    print("--- New Commit Received ---")
    print(json.dumps(new_commit_data, indent=2))
    print("---------------------------\n")

    # Load Model and Features
    model, features_list = load_prediction_assets()
    
    #Prepare Input Data for Model
    processed_data = prepare_input_data(new_commit_data, features_list)

    # Make Prediction
    prediction = model.predict(processed_data)[0]
    
    #gives the confidence score (e.g., [0.9, 0.1])
    prediction_proba = model.predict_proba(processed_data)[0]
    
    # Recall: 1 = failure, 0 = success
    
    print("--- AI Model Prediction ---")
    print(f"Model Prediction: {'FAILURE (1)' if prediction == 1 else 'SUCCESS (0)'}")
    print(f"Confidence (Failure): {prediction_proba[1]:.2%}")
    print(f"Confidence (Success): {prediction_proba[0]:.2%}\n")

    # AIOps pipeline.
    if prediction == 1:
        # Model predicts FAILURE.
        exit_code = fail_fast()
        sys.exit(exit_code)
    else:
        # Model predicts SUCCESS.
        exit_code = run_long_test_suite()
        sys.exit(exit_code)