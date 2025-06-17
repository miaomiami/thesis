import os
import pandas as pd

CSV_PATH = "segments/data.csv"
SEGMENT_DIR = "segments"

df = pd.read_csv(CSV_PATH)
renamed = 0
missing = 0

for new_path in df["path"]:
    new_filename = os.path.basename(new_path)
    
    # 找出所有旧文件中包含 new_filename 的结尾部分（为了匹配旧的冗余名）
    candidates = [f for f in os.listdir(SEGMENT_DIR) if f.endswith(new_filename)]
    
    if len(candidates) == 1:
        old_path = os.path.join(SEGMENT_DIR, candidates[0])
        new_path_full = os.path.join(SEGMENT_DIR, new_filename)
        
        if old_path != new_path_full:
            os.rename(old_path, new_path_full)
            renamed += 1
    elif len(candidates) == 0:
        print(f"[❌ 找不到匹配音频] {new_filename}")
        missing += 1
    else:
        print(f"[⚠️ 多个候选] {new_filename} 匹配到多个：{candidates}")

print(f"\n✅ 已成功重命名 {renamed} 个音频文件。")
print(f"❌ 未找到的文件数量：{missing}")
