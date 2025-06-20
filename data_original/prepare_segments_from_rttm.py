import os
import csv
import glob
from pathlib import Path
import subprocess

# Configuration parameters
RTTM_DIR = "./aishell4_data/rttm"      # .rttm file path
AUDIO_DIR = "./aishell4_data/wav"      # Original .flac audio path
OUTPUT_AUDIO_DIR = "./segments"        # Save the cut fragments
CSV_PATH = "./data.csv"                # Output CSV file
MIN_DURATION = 1.5                     # Minimum cut duration (in seconds)
TARGET_SR = 16000                      # Output sampling rate
AUDIO_EXT = ".wav"                     # Output audio format (.wav or .flac)

# Create Output Folder
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# Initialize CSV
csvfile = open(CSV_PATH, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["path", "label"])

# Parse rttm and cut audio
for rttm_file in glob.glob(os.path.join(RTTM_DIR, "*.rttm")):
    with open(rttm_file, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            basename = parts[1]             # Audio file name (without extension)
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]

            # Constructing a globally unique label
            label = f"{meeting_id}_{speaker}"

            # Filter out paragraphs that are too short
            if duration < MIN_DURATION:
                continue

            input_audio = os.path.join(AUDIO_DIR, basename + ".flac")
            output_name = f"{speaker}_{i:04d}{AUDIO_EXT}"
            output_path = os.path.join(OUTPUT_AUDIO_DIR, output_name)

            # ffmpeg cropping command
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_audio,
                "-ss", f"{start:.2f}",
                "-to", f"{end:.2f}",
                "-ar", str(TARGET_SR),
                "-ac", "1",
                output_path
            ]
            try:
                subprocess.run(command, check=True)
                writer.writerow([output_path, speaker])
            except subprocess.CalledProcessError:
                print(f"[Warning] Failed to extract: {output_path}")

csvfile.close()
print("The audio segmentation is complete and the CSV file has been saved to:", CSV_PATH)
