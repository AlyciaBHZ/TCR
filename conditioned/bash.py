"""
example script to run/submit the script to server
"""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of scripts to run
scripts = [
    # GIL
    ["python", "aa_predict.py", "-c", "1", "-i", "3", "-m", "[1,2,3,4,5,6,7,8,9,10,11]"],
    ["python", "aa_predict.py", "-c", "2", "-i", "3", "-m", "[1,2,3,4,5,6,7,8,9,10,11]"],
    ["python", "aa_predict.py", "-c", "3", "-i", "3", "-m", "[1,2,3,4,5,6,7,8,9,10,11]"],
    ["python", "aa_predict.py", "-c", "4", "-i", "3", "-m", "[1,2,3,4,5,6,7,8,9,10,11]"],
    ["python", "aa_predict.py", "-c", "5", "-i", "3", "-m", "[1,2,3,4,5,6,7,8,9,10,11]"],
    ["python", "aa_predict.py", "-c", "6", "-i", "3", "-m", "[1,2,3,4,5,6,7,8,9,10,11]"],
 # NIL
    ["python", "aa_predict.py", "-c", "1", "-i", "1", "-m", "[2,3,4,5,6,7,8,9,10,11,12]"],
    ["python", "aa_predict.py", "-c", "2", "-i", "1", "-m", "[2,3,4,5,6,7,8,9,10,11,12]"],
    ["python", "aa_predict.py", "-c", "3", "-i", "1", "-m", "[2,3,4,5,6,7,8,9,10,11,12]"],
    ["python", "aa_predict.py", "-c", "4", "-i", "1", "-m", "[2,3,4,5,6,7,8,9,10,11,12]"],
    ["python", "aa_predict.py", "-c", "5", "-i", "1", "-m", "[2,3,4,5,6,7,8,9,10,11,12]"],
    ["python", "aa_predict.py", "-c", "6", "-i", "1", "-m", "[2,3,4,5,6,7,8,9,10,11,12]"],
    #GLC
    ["python", "aa_predict.py", "-c", "1", "-i", "2", "-m", "[2,3,4,5,6,7,8,9,10,11,12,13]"],
    ["python", "aa_predict.py", "-c", "2", "-i", "2", "-m", "[2,3,4,5,6,7,8,9,10,11,12,13]"],
    ["python", "aa_predict.py", "-c", "3", "-i", "2", "-m", "[2,3,4,5,6,7,8,9,10,11,12,13]"],
    ["python", "aa_predict.py", "-c", "4", "-i", "2", "-m", "[2,3,4,5,6,7,8,9,10,11,12,13]"],
    ["python", "aa_predict.py", "-c", "5", "-i", "2", "-m", "[2,3,4,5,6,7,8,9,10,11,12,13]"],
    ["python", "aa_predict.py", "-c", "6", "-i", "2", "-m", "[2,3,4,5,6,7,8,9,10,11,12,13]"],
]

# Timeout duration (3 hours in seconds)
timeout_duration = 2 * 60 * 60  # 3 hours

# Function to run each script
def run_script(script, script_index):
    print(f"Starting script {script_index}: {' '.join(script)}")
    try:
        start_time = time.time()
        subprocess.run(script, timeout=timeout_duration, check=True)
        elapsed_time = time.time() - start_time
        print(f"Script {script_index} completed successfully in {elapsed_time:.2f} seconds.")
    except subprocess.TimeoutExpired:
        print(f"Script {script_index} timed out after {timeout_duration / 60:.2f} minutes and was terminated.")
    except subprocess.CalledProcessError as e:
        print(f"Script {script_index} exited with an error: {e}")
    print(f"Script {script_index} finished.\n")

# Use ThreadPoolExecutor to run 2-3 scripts concurrently
max_workers = 2  # Run up to 3 scripts at the same time

print("Starting concurrent execution...\n")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all scripts to the executor
    futures = {executor.submit(run_script, script, i + 1): i + 1 for i, script in enumerate(scripts)}

    # Monitor the progress of scripts
    for future in as_completed(futures):
        script_index = futures[future]
        try:
            future.result()  # Will raise exceptions if any occurred
        except Exception as e:
            print(f"An error occurred while running script {script_index}: {e}")

print("All scripts executed.")
