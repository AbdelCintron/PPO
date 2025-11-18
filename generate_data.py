import pandas as pd
import random
import time

#Main Configuration 
NUM_ROWS = 100
AUTHORS = ['main_dev', 'main_dev', 'main_dev', 'new_dev', 'contractor']
FILE_TYPES = ['.py', '.js', '.md', '.yml', '.css']

data = []

print(f"Generating {NUM_ROWS} rows of data...")

for _ in range(NUM_ROWS): 
    author = random.choice(AUTHORS)
    files_changed = random.randint(1, 20)
    lines_added = random.randint(0, 100)
    lines_deleted = random.randint(0, 50)

    #Simulating mix of files types being changed
    dominant_file_type = random.choice(FILE_TYPES)

    #Target Build

    fail_probability = 0.1

    if author in ['new_dev', 'contractor']:
        fail_probability += 0.3
    if files_changed > 15:
        fail_probability += 0.2
    if dominant_file_type == '.yml':
        fail_probability += 0.4
    if lines_added > 80:
        fail_probability += 0.1

    #Clamp Probability
    fail_probability = min(fail_probability, 0.9)

    build_status = random.uniform(1.0, 5.0) if random.random() < fail_probability else 'success'

    data.append({
        'author': author,
        'files_changed': files_changed,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'dominant_file_type': dominant_file_type,
        'build_status': build_status
    })

    df = pd.DataFrame(data)
    df.to_csv('build_history.csv', index=False)

    print(f"Successfully generated 'build_history.csv' with {len(df)} rows.")
    print("\n--- Data Head --- \n")
    print(df.head())