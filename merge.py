# ==========================================
# 📦 STEP 1: IMPORTS
# ==========================================
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ==========================================
# 📂 STEP 2: FILE PATHS
# ==========================================
DATASETS = {
    "BasicVNDN": "BasicVNDN-0.vec",
    "Benign": "BenignTraffic-0.vec",
    "Enhanced": "EnhancedAljubail-0.vec",
    "Flooding": "Attack01_InterestFlooding-0.vec",
    "Poisoning": "Attack02_ContentPoisoning-0.vec",
    "CachePollution": "Attack03_CachePollution-0.vec",
    "Hijacking": "Attack05_NamePrefixHijacking-0.vec"
}


# ==========================================
# 🏷️ STEP 3: LABELS
# ==========================================
LABEL_MAP = {
    "BasicVNDN": 0,
    "Benign": 0,
    "Enhanced": 0,
    "Flooding": 1,
    "Poisoning": 2,
    "CachePollution": 3,
    "Hijacking": 4
}


# ==========================================
# 🔥 IMPORTANT FILTER (CRITICAL)
# ==========================================
IMPORTANT_KEYWORDS = [
    "Interest", "Data", "Delay",
    "Drop", "Retry", "Cache",
    "Hit", "Miss"
]


# ==========================================
# 🔍 STEP 4: FIXED PARSER (FAST + SAFE)
# ==========================================
def parse_vec(file_path, sample_rate=20):
    vectors = {}
    data = defaultdict(list)

    counter = 0

    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Parsing {os.path.basename(file_path)}"):
            line = line.strip()

            if not line:
                continue

            # ----------------------------
            # VECTOR HEADER
            # ----------------------------
            if line.startswith("vector"):
                parts = line.split()

                if len(parts) >= 4:
                    vec_id = parts[1]

                    # CLEAN NAME
                    vec_name = parts[2] + "_" + parts[3]

                    # 🔥 FILTER IMPORTANT FEATURES ONLY
                    if not any(k.lower() in vec_name.lower() for k in IMPORTANT_KEYWORDS):
                        continue

                    vectors[vec_id] = vec_name

            # ----------------------------
            # DATA LINE (WITH SAMPLING)
            # ----------------------------
            elif line[0].isdigit():
                counter += 1

                # 🔥 SAMPLE DATA (VERY IMPORTANT)
                if counter % sample_rate != 0:
                    continue

                parts = line.split()

                if len(parts) >= 3:
                    vec_id = parts[0]

                    if vec_id in vectors:
                        try:
                            time = float(parts[1])
                            value = float(parts[2])

                            data[vectors[vec_id]].append((time, value))
                        except:
                            continue

    return data


# ==========================================
# 🔄 STEP 5: CONVERT TO DATAFRAME
# ==========================================
def to_dataframe(data):
    dfs = []

    for key, values in data.items():
        if len(values) == 0:
            continue

        df = pd.DataFrame(values, columns=["time", key])
        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    df_final = dfs[0]

    for df in dfs[1:]:
        df_final = pd.merge(df_final, df, on="time", how="outer")

    df_final = df_final.sort_values("time")

    # Forward fill + fill remaining
    df_final = df_final.ffill().fillna(0)

    return df_final


# ==========================================
# 🧩 STEP 6: PROCESS DATASETS
# ==========================================
all_datasets = []

for name, path in tqdm(DATASETS.items(), desc="Processing datasets"):
    print(f"\n🔹 Processing: {name}")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    raw_data = parse_vec(path)

    print(f"Total vectors found: {len(raw_data)}")

    if len(raw_data) == 0:
        print("⚠️ No useful vectors found (after filtering)")
        continue

    print("Sample vectors:", list(raw_data.keys())[:5])

    df = to_dataframe(raw_data)

    if df.empty:
        print("⚠️ Empty dataframe after processing")
        continue

    df["scenario"] = name
    df["label"] = LABEL_MAP[name]

    print(f"✅ Shape: {df.shape}")

    all_datasets.append(df)


# ==========================================
# 🔗 STEP 7: MERGE
# ==========================================
if len(all_datasets) == 0:
    raise ValueError("❌ No datasets processed! Check parsing.")

final_dataset = pd.concat(all_datasets, ignore_index=True)


# ==========================================
# 🧹 STEP 8: CLEANING
# ==========================================
final_dataset = final_dataset.drop_duplicates()
final_dataset = final_dataset.reset_index(drop=True)


# ==========================================
# 💾 STEP 9: SAVE
# ==========================================
output_file = "merged_vec_dataset.csv"
final_dataset.to_csv(output_file, index=False)

print("\n======================================")
print("🎉 MERGING COMPLETE")
print("Final Shape:", final_dataset.shape)
print("Saved as:", output_file)
print("======================================")