import os
import pandas as pd

CSV_PATH = "segments/data.csv"
SEGMENT_DIR = "segments"

df = pd.read_csv(CSV_PATH)
renamed = 0
missing = 0

for new_path in df["path"]:
    new_filename = os.path.basename(new_path)
    
    # Find all old files that have new_filename at the end (to match the old redundant name)
    candidates = [f for f in os.listdir(SEGMENT_DIR) if f.endswith(new_filename)]
    
    if len(candidates) == 1:
        old_path = os.path.join(SEGMENT_DIR, candidates[0])
        new_path_full = os.path.join(SEGMENT_DIR, new_filename)
        
        if old_path != new_path_full:
            os.rename(old_path, new_path_full)
            renamed += 1
    elif len(candidates) == 0:
        print(f"[No matching audio found] {new_filename}")
        missing += 1
    else:
        print(f"[Multiple candidates] {new_filename} matches multiple：{candidates}")

print(f"\n {renamed} audio files renamed successfully。")
print(f"Number of files not found：{missing}")
