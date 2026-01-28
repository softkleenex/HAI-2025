import argparse
import os
from datetime import datetime

def log_submission(score, filename, model_name, note=""):
    history_file = "SUBMISSION_HISTORY.md"
    
    # Check if file exists to determine ID
    last_id = 0
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "|" in line and line.split("|")[1].strip().isdigit():
                    last_id = int(line.split("|")[1].strip())
    
    new_id = last_id + 1
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Markdown Table Row format
    # | ID | Date | File Name | Score (AUC) | Model | Note |
    new_row = f"| {new_id} | {date_str} | {filename} | {score} | {model_name} | {note} |\n"
    
    # Append
    with open(history_file, 'a') as f:
        f.write(new_row)
        
    print(f"Logged submission {new_id}: Score {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=float, required=True, help='Leaderboard Score')
    parser.add_argument('--file', type=str, required=True, help='Submission filename')
    parser.add_argument('--model', type=str, required=True, help='Model name used')
    parser.add_argument('--note', type=str, default="", help='Extra notes')
    args = parser.parse_args()
    
    log_submission(args.score, args.file, args.model, args.note)
