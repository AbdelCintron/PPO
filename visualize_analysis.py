import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

print("--- Starting Visual Analysis ---")

# --- 0. Setup Output Directory ---
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}/")

# --- 1. Load Assets ---
try:
    df = pd.read_csv('build_history.csv')
    model = joblib.load('ci_model.pkl')
    with open('model_features.json', 'r') as f:
        feature_names = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Make sure you have run 'generate_data.py' and 'train_model.py' first.")
    exit()

# --- 2. Visual 1: Failure Rate by Author (EDA) ---
# This shows the raw data bias before any AI is involved
print("Generating Chart 1: Failure Rate by Author...")
plt.figure(figsize=(10, 6))

# Calculate failure rates
author_stats = df.groupby('author')['build_status'].value_counts(normalize=True).unstack().fillna(0)
if 'failure' in author_stats.columns:
    # Sort by failure rate descending
    author_stats['failure'].sort_values(ascending=False).plot(kind='bar', color='salmon', edgecolor='black')
    plt.title('Historical Failure Rate by Developer (Raw Data)')
    plt.ylabel('Failure Probability')
    plt.xlabel('Developer Persona')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'analysis_1_author_risk.png')
    plt.savefig(save_path)
    print(f"Saved '{save_path}'")

# --- 3. Visual 2: Feature Importance (AI Logic) ---
# This explains "What does the robot care about most?"
print("Generating Chart 2: Feature Importance...")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # Sort descending

plt.figure(figsize=(12, 6))
plt.title("What Drives the AI's Decision? (Feature Importance)")
plt.bar(range(len(importances)), importances[indices], color='skyblue', align='center', edgecolor='black')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Relative Importance')
plt.tight_layout()

save_path = os.path.join(output_dir, 'analysis_2_feature_importance.png')
plt.savefig(save_path)
print(f"Saved '{save_path}'")

# --- 4. Visual 3: Confusion Matrix (Model Accuracy) ---
# This visualizes "Where does the model get confused?"
print("Generating Chart 3: Confusion Matrix...")

# We need to regenerate the test set to show how it performs on unseen data
# (Replicating the logic from train_model.py)
df_processed = pd.get_dummies(df, columns=['author', 'dominant_file_type'], drop_first=True)
df_processed['build_status'] = df_processed['build_status'].map({'success': 0, 'failure': 1})
y = df_processed['build_status']
X = df_processed.drop('build_status', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Success', 'Failure'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Actual vs Predicted)")

save_path = os.path.join(output_dir, 'analysis_3_confusion_matrix.png')
plt.savefig(save_path)
print(f"Saved '{save_path}'")

# --- 5. Visual 4: Failure Rate Trend (Rolling Average) ---
print("Generating Chart 4: Failure Rate Trend...")

# Convert status to numeric for calculation: Failure=1, Success=0
# We use the raw df here
df['numeric_status'] = df['build_status'].map({'failure': 1, 'success': 0})

# Calculate Rolling Average (moving average over last 100 builds)
# This smoothes out the noise to show the underlying trend
window_size = 100
rolling_fail_rate = df['numeric_status'].rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))
plt.plot(df.index, rolling_fail_rate, color='purple', linewidth=2, label=f'Failure Rate ({window_size}-build avg)')
plt.title(f'Trend of Build Stability (Rolling {window_size}-Build Failure Rate)')
plt.xlabel('Build Number (Chronological Order)')
plt.ylabel('Failure Rate (0.0 = 0%, 1.0 = 100%)')

# Add a reference line for the overall average
overall_avg = df['numeric_status'].mean()
plt.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.5, label=f'Overall Average ({overall_avg:.1%})')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = os.path.join(output_dir, 'analysis_4_failure_trend.png')
plt.savefig(save_path)
print(f"Saved '{save_path}'")

print(f"\nAnalysis Complete. Check the 4 PNG files in the '{output_dir}/' folder.")