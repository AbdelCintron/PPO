import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# Removed LabelEncoder as we will use a safer manual map

print("Starting model training script...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('build_history.csv')
except FileNotFoundError:
    print("Error: 'build_history.csv' not found.")
    print("Please run 'python generate_data.py' first.")
    exit()

# --- 2. Feature Engineering & Preprocessing ---
categorical_features = ['author', 'dominant_file_type']
df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# --- Preprocess the Target Variable (FIXED) ---
# We explicitly map the strings to numbers. 
# This prevents errors if the data is corrupted.
target_map = {'success': 0, 'failure': 1}

print(f"Unique values found in build_status: {df['build_status'].unique()}")

# Map the values. Any value not in the map becomes NaN (empty)
df_processed['build_status'] = df_processed['build_status'].map(target_map)

# Check if we have any bad data (NaNs)
if df_processed['build_status'].isnull().any():
    print("\n🚨 ERROR: 'build_status' column contains invalid data!")
    print("It must only contain 'success' or 'failure'.")
    print("Please delete 'build_history.csv' and run 'python generate_data.py' again.")
    exit()

print("\n--- Preprocessed Data Head ---")
print(df_processed.head())

# --- 3. Define Features (X) and Target (y) ---
target_column = 'build_status'
y = df_processed[target_column]
X = df_processed.drop(target_column, axis=1)

# --- CRITICAL MLOps Step ---
final_features_list = list(X.columns)
print(f"\nModel will be trained on {len(final_features_list)} features.")

# --- 4. Split Data ---
# Now that we are sure 'y' only contains 0s and 1s, this will work.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Train Model ---
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- 6. Evaluate Model ---
print("\n--- Model Evaluation (on Test Set) ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['success (0)', 'failure (1)']))

# --- 7. Save Model ---
model_filename = 'ci_model.pkl'
features_filename = 'model_features.json'

print(f"\nSaving trained model to {model_filename}")
joblib.dump(model, model_filename)

print(f"Saving feature list to {features_filename}")
with open(features_filename, 'w') as f:
    json.dump(final_features_list, f)

print("\nTraining complete. Model artifacts are ready.")