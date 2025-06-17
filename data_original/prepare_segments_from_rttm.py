import os
import csv
import glob
from pathlib import Path
import subprocess

# === 配置参数 ===
RTTM_DIR = "./aishell4_data/rttm"      # .rttm 文件路径
AUDIO_DIR = "./aishell4_data/wav"      # 原始 .flac 音频路径
OUTPUT_AUDIO_DIR = "./segments"        # 保存切好的片段
CSV_PATH = "./data.csv"                # 输出 CSV 文件
MIN_DURATION = 1.5                     # 最短切段时长（单位秒）
TARGET_SR = 16000                      # 输出采样率
AUDIO_EXT = ".wav"                     # 输出音频格式（.wav 或 .flac）

# === 创建输出文件夹 ===
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# === 初始化 CSV ===
csvfile = open(CSV_PATH, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["path", "label"])

# === 解析 rttm 并切音频 ===
for rttm_file in glob.glob(os.path.join(RTTM_DIR, "*.rttm")):
    with open(rttm_file, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            basename = parts[1]             # 音频文件名（无扩展名）
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]

            # ✅ 构造全局唯一的标签
            label = f"{meeting_id}_{speaker}"

            # 过滤掉太短的段落
            if duration < MIN_DURATION:
                continue

            input_audio = os.path.join(AUDIO_DIR, basename + ".flac")
            output_name = f"{speaker}_{i:04d}{AUDIO_EXT}"
            output_path = os.path.join(OUTPUT_AUDIO_DIR, output_name)

            # ffmpeg 裁剪命令
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
print("✅ 音频切段完成，CSV文件已保存至：", CSV_PATH)
