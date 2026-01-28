import os
import time
import subprocess
import sys

def is_running(process_name):
    try:
        # Check if any process has 'process_name' in its command line
        output = subprocess.check_output(['pgrep', '-f', process_name])
        return len(output) > 0
    except subprocess.CalledProcessError:
        return False

def run_command(cmd):
    print(f"Executing: {cmd}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(cmd, shell=True, env=env)
    return result.returncode == 0

def main():
    target = "src/trainer/train.py"
    print(f"Waiting for {target} to finish...")
    
    # Wait for the training to start if it hasn't yet (briefly)
    time.sleep(10)
    
    while is_running(target):
        # Check every 2 minutes
        time.sleep(120)
    
    print(f"{target} has finished. Starting automated inference and submission fix.")
    
    # Step 1: Inference
    inference_cmd = "python src/trainer/inference.py --config configs/base_config.yaml --checkpoint checkpoints/xception_strong_aug_best.pt --output data/submission/submission_xception_final.csv"
    if run_command(inference_cmd):
        print("Inference successful.")
    else:
        print("Inference failed.")
        return

    # Step 2: Fix Submission
    fix_cmd = "python scripts/fix_submission.py --input data/submission/submission_xception_final.csv --output data/submission/final_fixed_xception_strong_aug.csv"
    if run_command(fix_cmd):
        print("Submission fix successful.")
        print("Final File Ready: data/submission/final_fixed_xception_strong_aug.csv")
    else:
        print("Submission fix failed.")

if __name__ == "__main__":
    main()
