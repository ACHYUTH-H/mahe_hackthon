# ==========================================
# 📦 IMPORTS
# ==========================================
import os
import pandas as pd
import numpy as np
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

# ==========================================
# 📂 LOAD DATA
# ==========================================
print("\n=== LOADING DATA ===")
df = pd.read_csv("final_features.csv")
df = df.fillna(0).replace([np.inf, -np.inf], 0)

X = df.drop(columns=["label", "scenario"])
y = df["label"]

# ==========================================
# SCALE FEATURES
# ==========================================
print("\n=== SCALING FEATURES ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 🚀 STAGE 1 → BINARY CLASSIFIER
# ==========================================
print("\n=== STAGE 1 TRAINING (Binary) ===")

y_binary = (y != 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.3, stratify=y_binary, random_state=42
)

stage1 = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss"
)

stage1.fit(X_train, y_train)

print("\nStage 1 Results:")
print(classification_report(y_test, stage1.predict(X_test)))

# ==========================================
# 🚀 STAGE 2 → ATTACK CLASSIFIER
# ==========================================
print("\n=== STAGE 2 TRAINING (Attack Classification) ===")

attack_df = df[df["label"] != 0]

X_attack = attack_df.drop(columns=["label", "scenario"])
y_attack_raw = attack_df["label"]

# Encode labels
le = LabelEncoder()
y_attack = le.fit_transform(y_attack_raw)

X_attack_scaled = scaler.transform(X_attack)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_attack_scaled, y_attack, test_size=0.3, stratify=y_attack, random_state=42
)

# ==========================================
# FEATURE SELECTION
# ==========================================
print("\n=== FEATURE SELECTION ===")

selector = SelectFromModel(
    xgb.XGBClassifier(n_estimators=200, max_depth=5),
    threshold="median"
)

selector.fit(X_train2, y_train2)

X_train2 = selector.transform(X_train2)
X_test2 = selector.transform(X_test2)

print("Selected features:", X_train2.shape[1])

# ==========================================
# CLASS WEIGHTS (BALANCED)
# ==========================================
class_counts = Counter(y_train2)
total = sum(class_counts.values())

weights = {
    c: (total / (len(class_counts) * cnt)) * (
        1.8 if le.inverse_transform([c])[0] == 1 else
        1.4 if le.inverse_transform([c])[0] == 3 else
        1.6 if le.inverse_transform([c])[0] == 4 else
        1
    )
    for c, cnt in class_counts.items()
}

sample_weights = np.array([weights[label] for label in y_train2])

# ==========================================
# TRAIN STAGE 2 MODEL
# ==========================================
print("\n=== TRAINING STAGE 2 MODEL ===")

stage2 = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    reg_lambda=1.5,
    reg_alpha=0.5,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss"
)

stage2.fit(X_train2, y_train2, sample_weight=sample_weights)

# ==========================================
# THRESHOLD TUNING (MANUAL)
# ==========================================
thresholds = {
    0: 0.45,  # Flooding
    1: 0.5,   # Cache
    2: 0.4    # Hijacking
}

# ==========================================
# STAGE 2 EVALUATION (THRESHOLD BASED)
# ==========================================
print("\n=== STAGE 2 RESULTS (THRESHOLD) ===")

probs = stage2.predict_proba(X_test2)
final_preds = []

for p in probs:
    best_class = None
    best_score = -1

    for i in range(len(p)):
        score = p[i] / thresholds[i]
        if score > best_score:
            best_score = score
            best_class = i

    final_preds.append(best_class)

decoded_preds = le.inverse_transform(final_preds)
decoded_true = le.inverse_transform(y_test2)

print(classification_report(decoded_true, decoded_preds))

# ==========================================
# FINAL PIPELINE EVALUATION
# ==========================================
print("\n=== FINAL PIPELINE ===")

final_preds_all = []

for i in range(len(X_scaled)):
    x = X_scaled[i].reshape(1, -1)

    if stage1.predict(x)[0] == 0:
        final_preds_all.append(0)
    else:
        x_sel = selector.transform(x)
        probs = stage2.predict_proba(x_sel)[0]

        best_class = None
        best_score = -1

        for j in range(len(probs)):
            score = probs[j] / thresholds[j]
            if score > best_score:
                best_score = score
                best_class = j

        pred = le.inverse_transform([best_class])[0]
        final_preds_all.append(pred)

print("\nFinal Results:")
print(classification_report(y, final_preds_all))

# ==========================================
# 📦 SAVE FINAL MODEL (IMPORTANT)
# ==========================================
print("\n=== SAVING FINAL MODEL ===")

os.makedirs("models", exist_ok=True)

final_model = {
    "stage1": stage1,
    "stage2": stage2,
    "scaler": scaler,
    "selector": selector,
    "label_encoder": le,
    "thresholds": thresholds
}

model_path = "models/final_rsu_model.pkl"
joblib.dump(final_model, model_path)

print(f"✅ Model saved at: {model_path}")