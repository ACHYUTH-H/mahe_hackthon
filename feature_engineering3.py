# ==========================================
# 📦 IMPORTS
# ==========================================
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("merged_vec_dataset.csv")
print("Loaded dataset:", df.shape)

# ==========================================
# HELPERS
# ==========================================
def safe_filter(df, keyword):
    cols = [c for c in df.columns if keyword.lower() in c.lower()]
    if len(cols) == 0:
        return pd.Series([0]*len(df))
    return df[cols].sum(axis=1)

def safe_corr(x, y=None):
    if y is None:
        if np.std(x) == 0:
            return 0
        return np.corrcoef(range(len(x)), x)[0,1]
    else:
        if np.std(x) == 0 or np.std(y) == 0:
            return 0
        return np.corrcoef(x, y)[0,1]

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(df, window_size=10):

    features = []

    for i in tqdm(range(window_size, len(df) - window_size), desc="Extracting features"):

        current = df.iloc[i:i+window_size]
        previous = df.iloc[i-window_size:i]

        feat = {}

        interest = safe_filter(current, "Interest").values
        data_pkt = safe_filter(current, "Data").values
        delay = safe_filter(current, "Delay").values
        drop = safe_filter(current, "Drop").values

        # ==================================
        # BASIC FEATURES
        # ==================================
        feat["interest_rate"] = np.mean(interest)
        feat["data_rate"] = np.mean(data_pkt)
        feat["delay_mean"] = np.mean(delay)
        feat["drop_rate"] = np.mean(drop)

        # ==================================
        # TEMPORAL FEATURES
        # ==================================
        feat["interest_diff"] = np.mean(np.diff(interest))
        feat["delay_diff"] = np.mean(np.diff(delay))

        feat["interest_acc"] = np.mean(np.diff(np.diff(interest)))
        feat["delay_acc"] = np.mean(np.diff(np.diff(delay)))

        feat["interest_peak_pos"] = np.argmax(interest) / len(interest)
        feat["delay_peak_pos"] = np.argmax(delay) / len(delay)

        feat["interest_spike_width"] = np.sum(interest > np.mean(interest))
        feat["delay_spike_width"] = np.sum(delay > np.mean(delay))

        feat["interest_roll_var"] = np.nan_to_num(
            pd.Series(interest).rolling(3).var().mean(), nan=0
        )
        feat["delay_roll_var"] = np.nan_to_num(
            pd.Series(delay).rolling(3).var().mean(), nan=0
        )

        feat["interest_trend_strength"] = safe_corr(interest)
        feat["delay_trend_strength"] = safe_corr(delay)

        # ==================================
        # 🔥 DISCRIMINATIVE FEATURES (1 vs 3)
        # ==================================
        feat["delay_growth_rate"] = (delay[-1] - delay[0]) / (len(delay)+1e-5)
        feat["delay_spike_ratio"] = np.max(delay) / (np.mean(delay)+1e-5)

        feat["interest_delay_corr"] = safe_corr(interest, delay)

        feat["high_interest_duration"] = np.sum(interest > np.percentile(interest, 75))
        feat["high_delay_duration"] = np.sum(delay > np.percentile(delay, 75))

        feat["delay_variation_ratio"] = np.std(delay) / (np.mean(delay)+1e-5)

        # ==================================
        # 🔥 CLASS 4 FEATURE (Hijacking)
        # ==================================
        feat["delay_instability"] = np.std(delay) * np.mean(np.abs(np.diff(delay)))

        # ==================================
        # 🔥 CLASS 1 FEATURE (Flooding)
        # ==================================
        feat["interest_to_delay_ratio"] = np.mean(interest) / (np.mean(delay) + 1e-5)

        # ==================================
        # CONTEXT FEATURES
        # ==================================
        prev_interest = np.mean(safe_filter(previous, "Interest"))
        prev_delay = np.mean(safe_filter(previous, "Delay"))

        feat["interest_rate_change"] = feat["interest_rate"] - prev_interest
        feat["delay_change"] = feat["delay_mean"] - prev_delay

        # ==================================
        # CONSISTENCY
        # ==================================
        feat["data_variance"] = np.var(data_pkt)
        feat["consistency_score"] = 1 / (np.var(data_pkt) + 1e-5)

        # ==================================
        # FINAL CLEANUP
        # ==================================
        for k in feat:
            if np.isnan(feat[k]) or np.isinf(feat[k]):
                feat[k] = 0

        # ==================================
        # LABELS
        # ==================================
        feat["label"] = df.iloc[i]["label"]
        feat["scenario"] = df.iloc[i]["scenario"]

        features.append(feat)

    return pd.DataFrame(features)

# ==========================================
# RUN
# ==========================================
feature_df = extract_features(df)

print("Feature dataset shape:", feature_df.shape)

# ==========================================
# SAVE
# ==========================================
feature_df.to_csv("final_features.csv", index=False)

print("Saved as final_features.csv")