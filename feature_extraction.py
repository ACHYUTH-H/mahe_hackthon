# ==========================================
# 📦 STEP 1: IMPORTS
# ==========================================
import pandas as pd
import numpy as np
from tqdm import tqdm


# ==========================================
# 📂 STEP 2: LOAD MERGED DATA
# ==========================================
INPUT_FILE = "merged_vec_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("Loaded dataset:", df.shape)


# ==========================================
# 🔧 STEP 3: HELPER FUNCTION
# ==========================================
def safe_filter(df, keyword):
    cols = [c for c in df.columns if keyword.lower() in c.lower()]
    if len(cols) == 0:
        return pd.Series([0]*len(df))
    return df[cols].sum(axis=1)


# ==========================================
# 🔥 STEP 4: CONTEXT-AWARE FEATURE EXTRACTION
# ==========================================
def extract_features(df, window_size=10):
    features = []

    for i in tqdm(range(window_size, len(df) - window_size), desc="Extracting features"):

        current = df.iloc[i:i+window_size]
        previous = df.iloc[i-window_size:i]

        feat = {}

        # =============================
        # 🔹 BASIC TRAFFIC FEATURES
        # =============================
        interest = safe_filter(current, "Interest")
        data_pkt = safe_filter(current, "Data")
        delay = safe_filter(current, "Delay")
        drops = safe_filter(current, "Drop")
        retry = safe_filter(current, "Retry")

        total_packets = interest + data_pkt + 1e-5

        feat["interest_rate"] = interest.mean()
        feat["data_rate"] = data_pkt.mean()
        feat["interest_data_ratio"] = interest.sum() / (data_pkt.sum() + 1e-5)

        # =============================
        # 🔹 RELIABILITY FEATURES
        # =============================
        feat["drop_rate"] = drops.sum() / (total_packets.sum() + 1e-5)
        feat["retry_rate"] = retry.sum() / window_size

        # =============================
        # 🔹 PERFORMANCE FEATURES
        # =============================
        feat["delay_mean"] = delay.mean()
        feat["delay_std"] = delay.std()
        feat["throughput"] = data_pkt.sum() / window_size

        # =============================
        # 🔹 CACHE FEATURES
        # =============================
        cache_hit = safe_filter(current, "Hit")
        cache_miss = safe_filter(current, "Miss")

        feat["cache_hit_ratio"] = cache_hit.sum() / (total_packets.sum() + 1e-5)
        feat["cache_miss_ratio"] = cache_miss.sum() / (total_packets.sum() + 1e-5)

        # =============================
        # 🔹 TEMPORAL CONTEXT
        # =============================
        prev_interest = safe_filter(previous, "Interest").mean()
        prev_delay = safe_filter(previous, "Delay").mean()

        feat["interest_rate_change"] = feat["interest_rate"] - prev_interest
        feat["delay_change"] = feat["delay_mean"] - prev_delay

        # =============================
        # 🔹 TREND FEATURES
        # =============================
        try:
            feat["interest_trend"] = np.polyfit(range(len(interest)), interest, 1)[0]
            feat["delay_trend"] = np.polyfit(range(len(delay)), delay, 1)[0]
        except:
            feat["interest_trend"] = 0
            feat["delay_trend"] = 0

        # =============================
        # 🔹 CONSISTENCY FEATURES
        # =============================
        feat["data_variance"] = data_pkt.var()
        feat["consistency_score"] = 1 / (data_pkt.var() + 1e-5)

        # =============================
        # 🔹 BURST DETECTION
        # =============================
        feat["burst_score"] = max(interest) / (np.mean(interest) + 1e-5)

        # =============================
        # 🔹 LABEL + SCENARIO
        # =============================
        feat["label"] = df.iloc[i]["label"]
        feat["scenario"] = df.iloc[i]["scenario"]

        features.append(feat)

    return pd.DataFrame(features)


# ==========================================
# ⚙️ STEP 5: RUN FEATURE EXTRACTION
# ==========================================
feature_df = extract_features(df, window_size=10)

print("Feature dataset shape:", feature_df.shape)


# ==========================================
# 💾 STEP 6: SAVE FEATURES
# ==========================================
OUTPUT_FILE = "final_features.csv"

feature_df.to_csv(OUTPUT_FILE, index=False)

print("Saved features to:", OUTPUT_FILE)