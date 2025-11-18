import pandas as pd
import random

# --- VERIFICATION PRINT ---
print("--- Running Data Generator ---")

# --- Configuration ---
NUM_ROWS = 10000
AUTHORS = ['main_dev', 'main_dev', 'main_dev', 'new_dev', 'contractor']
FILE_TYPES = ['.py', '.js', '.md', '.yml', '.css']

data = []

print(f"Generating {NUM_ROWS} rows of fake build data...")

for _ in range(NUM_ROWS):
    # 1. Generate Inputs
    author = random.choice(AUTHORS)
    files_changed = random.randint(1, 20)
    lines_added = random.randint(0, 100)
    lines_deleted = random.randint(0, 50)
    dominant_file_type = random.choice(FILE_TYPES)
    
    # 2. Calculate Failure Probability (The Logic)
    fail_probability = 0.1
    if author in ['new_dev', 'contractor']:
        fail_probability += 0.3
    if files_changed > 15:
        fail_probability += 0.2
    if dominant_file_type == '.yml':
        fail_probability += 0.4
    if lines_added > 80:
        fail_probability += 0.1
        
    fail_probability = min(fail_probability, 0.9)
    
    # 3. Determine Status
    # STRICTLY set this to string 'success' or 'failure'
    is_fail = random.random() < fail_probability
    build_status = 'failure' if is_fail else 'success'
    
    # 4. Append to list
    # We explicitly do NOT include build_time here to avoid confusion
    data.append({
        'author': author,
        'files_changed': files_changed,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'dominant_file_type': dominant_file_type,
        'build_status': build_status 
    })

# 5. Create DataFrame
df = pd.DataFrame(data)

# 6. VERIFICATION STEP (Prints to console before saving)
print("\n--- Verifying Data Structure ---")
unique_values = df['build_status'].unique()
print(f"Unique values in 'build_status': {unique_values}")

# Sanity check
if len(unique_values) > 2 or not all(x in ['success', 'failure'] for x in unique_values):
    print("ERROR: Data generation failed. Unknown values detected.")
else:
    # Only save if verification passes
    df.to_csv('build_history.csv', index=False)
    print("Verification passed. 'build_history.csv' saved successfully. \n")
    print(df.head())