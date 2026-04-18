# ==========================================
# 📦 STEP 1: IMPORTS
# ==========================================
import pandas as pd
import numpy as np
from tqdm import tqdm


# ==========================================
# 📂 STEP 2: LOAD DATA
# ==========================================
df = pd.read_csv("merged_vec_dataset.csv")
print("Loaded dataset:", df.shape)


# ==========================================
# 🔧 HELPER FUNCTIONS
# ==========================================
def safe_filter(df, keyword):
    cols = [c for c in df.columns if keyword.lower() in c.lower()]
    if len(cols) == 0:
        return pd.Series([0]*len(df))
    return df[cols].sum(axis=1)


# ==========================================
# 🔥 STEP 3: FEATURE EXTRACTION
# ==========================================
def extract_features(df, window_size=10):

    features = []

    for i in tqdm(range(window_size, len(df) - window_size), desc="Extracting features"):

        current = df.iloc[i:i+window_size]
        previous = df.iloc[i-window_size:i]

        feat = {}

        # =============================
        # 🔹 BASIC TRAFFIC
        # =============================
        interest = safe_filter(current, "Interest").values
        data_pkt = safe_filter(current, "Data").values
        delay = safe_filter(current, "Delay").values
        drops = safe_filter(current, "Drop").values
        retry = safe_filter(current, "Retry").values

        total_packets = interest + data_pkt + 1e-5

        feat["interest_rate"] = np.mean(interest)
        feat["data_rate"] = np.mean(data_pkt)
        feat["interest_data_ratio"] = np.sum(interest) / (np.sum(data_pkt) + 1e-5)

        # =============================
        # 🔹 RELIABILITY
        # =============================
        feat["drop_rate"] = np.sum(drops) / (np.sum(total_packets) + 1e-5)
        feat["retry_rate"] = np.sum(retry) / window_size

        # =============================
        # 🔹 PERFORMANCE
        # =============================
        feat["delay_mean"] = np.mean(delay)
        feat["delay_std"] = np.std(delay)
        feat["throughput"] = np.sum(data_pkt) / window_size

        # =============================
        # 🔹 CACHE
        # =============================
        cache_hit = safe_filter(current, "Hit").values
        cache_miss = safe_filter(current, "Miss").values

        feat["cache_hit_ratio"] = np.sum(cache_hit) / (np.sum(total_packets) + 1e-5)
        feat["cache_miss_ratio"] = np.sum(cache_miss) / (np.sum(total_packets) + 1e-5)

        # =============================
        # 🔥 NEW ADVANCED FEATURES
        # =============================

        # ✅ 1. ENTROPY (VERY IMPORTANT)
        prob = data_pkt / (np.sum(data_pkt) + 1e-5)
        feat["packet_entropy"] = -np.sum(prob * np.log(prob + 1e-5))

        # ✅ 2. BURSTINESS
        feat["burst_ratio"] = np.max(interest) / (np.mean(interest) + 1e-5)
        feat["burst_variance"] = np.var(interest)

        # ✅ 3. DELAY SHAPE
        feat["delay_skew"] = pd.Series(delay).skew()
        feat["delay_kurtosis"] = pd.Series(delay).kurt()

        # ✅ 4. TEMPORAL INSTABILITY
        feat["interest_instability"] = np.std(np.diff(interest))
        feat["delay_instability"] = np.std(np.diff(delay))

        # ✅ 5. CACHE DYNAMICS
        feat["cache_change_rate"] = np.mean(np.diff(cache_hit))

        # =============================
        # 🔹 TEMPORAL CONTEXT
        # =============================
        prev_interest = np.mean(safe_filter(previous, "Interest"))
        prev_delay = np.mean(safe_filter(previous, "Delay"))

        feat["interest_rate_change"] = feat["interest_rate"] - prev_interest
        feat["delay_change"] = feat["delay_mean"] - prev_delay

        # =============================
        # 🔹 TREND
        # =============================
        try:
            feat["interest_trend"] = np.polyfit(range(len(interest)), interest, 1)[0]
            feat["delay_trend"] = np.polyfit(range(len(delay)), delay, 1)[0]
        except:
            feat["interest_trend"] = 0
            feat["delay_trend"] = 0

        # =============================
        # 🔹 CONSISTENCY
        # =============================
        feat["data_variance"] = np.var(data_pkt)
        feat["consistency_score"] = 1 / (np.var(data_pkt) + 1e-5)

        # =============================
        # 🔹 LABEL
        # =============================
        feat["label"] = df.iloc[i]["label"]
        feat["scenario"] = df.iloc[i]["scenario"]

        features.append(feat)

    return pd.DataFrame(features)


# ==========================================
# ⚙️ STEP 4: RUN
# ==========================================
feature_df = extract_features(df, window_size=10)

print("Feature dataset shape:", feature_df.shape)


# ==========================================
# 💾 STEP 5: SAVE
# ==========================================
feature_df.to_csv("final_features.csv", index=False)

print("Saved as final_features.csv")