import os
import sys
import time
import subprocess
import signal

def run_command_background(cmd, log_file):
    """Run command with nohup in background and return PID"""
    full_cmd = f"nohup {cmd} > {log_file} 2>&1 & echo $!"
    process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = int(process.stdout.read().decode().strip())
    print(f"Started: {cmd} (PID: {pid}) -> Logging to {log_file}")
    return pid

def is_process_running(pid):
    """Check if PID is running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def wait_for_process(pid, name):
    print(f"Waiting for {name} (PID: {pid}) to finish...")
    start_time = time.time()
    while is_process_running(pid):
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\rElapsed: {elapsed}s")
        sys.stdout.flush()
        time.sleep(60) # Check every 1 min
    print(f"\n{name} finished!")

def main():
    # 1. Start Training (DINO Large Fine-tuning)
    train_cmd = "env PYTHONPATH=. python src/trainer/train.py --config configs/dino_large.yaml"
    train_log = "logs/train_dino_large.log"
    train_pid = run_command_background(train_cmd, train_log)
    
    # Wait for training
    wait_for_process(train_pid, "Training")
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/dino_large_finetune_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Training failed?")
        return

    # 2. Inference
    print("Starting Inference...")
    inference_cmd = f"env PYTHONPATH=. python src/trainer/inference.py --config configs/dino_large.yaml --checkpoint {checkpoint_path} --output data/submission/submission_dino_large.csv"
    os.system(inference_cmd)
    
    if not os.path.exists("data/submission/submission_dino_large.csv"):
        print("Error: Inference failed.")
        return

    # 3. Fix Submission
    print("Fixing Submission format...")
    fix_cmd = "python scripts/fix_submission.py --input data/submission/submission_dino_large.csv --output data/submission/final_dino_large_finetuned.csv"
    os.system(fix_cmd)
    
    print("Pipeline Complete! File: data/submission/final_dino_large_finetuned.csv")

if __name__ == "__main__":
    main()
