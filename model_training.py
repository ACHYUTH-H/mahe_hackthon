# ==========================================
# 📦 STEP 1: IMPORTS
# ==========================================
import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import xgboost as xgb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


# ==========================================
# 📝 STEP 2: LOGGING
# ==========================================
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)


# ==========================================
# ⏳ TQDM CALLBACK
# ==========================================
class TqdmCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = tqdm(total=1, desc=f"Epoch {epoch+1}", leave=False)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.close()


# ==========================================
# 📂 STEP 3: LOAD DATA
# ==========================================
log("Loading dataset...")
df = pd.read_csv("final_features.csv")

log(f"Dataset loaded: {df.shape}")

X = df.drop(columns=["label", "scenario"])
y = df["label"]


# ==========================================
# 🔥 STEP 4: LABEL ENCODING (CRITICAL FIX)
# ==========================================
le = LabelEncoder()
y = le.fit_transform(y)

joblib.dump(le, "label_encoder.pkl")
log(f"Labels encoded: {np.unique(y)}")


# ==========================================
# 🔧 STEP 5: SCALING
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")
log("Scaler saved.")


# ==========================================
# 📊 STEP 6: SPLIT (70/15/15)
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# ==========================================
# 🔥 STEP 7: AUTOENCODER
# ==========================================
X_train_normal = X_train[y_train == 0]
X_val_normal = X_val[y_val == 0]

input_dim = X_train.shape[1]

inp = Input(shape=(input_dim,))
x = Dense(64, activation="relu")(inp)
x = Dense(32, activation="relu")(x)
encoded = Dense(16, activation="relu")(x)

x = Dense(32, activation="relu")(encoded)
x = Dense(64, activation="relu")(x)
out = Dense(input_dim)(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer=Adam(1e-3), loss="mse")

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("autoencoder_best.h5", save_best_only=True),
    TqdmCallback()
]

log("Training Autoencoder...")
autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=20,
    batch_size=64,
    validation_data=(X_val_normal, X_val_normal),
    callbacks=callbacks,
    verbose=0
)

autoencoder.save("autoencoder_final.h5")
log("Autoencoder saved.")


# ==========================================
# 🔥 STEP 8: LSTM (FIXED)
# ==========================================
def create_sequences(X, y, seq_len=10):
    Xs, ys = [], []
    for i in tqdm(range(len(X) - seq_len), desc="Creating sequences"):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)


log("Generating sequences...")
X_seq, y_seq = create_sequences(X_scaled, y)

X_train_s, X_temp_s, y_train_s, y_temp_s = train_test_split(
    X_seq, y_seq, test_size=0.3, stratify=y_seq, random_state=42
)

X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
    X_temp_s, y_temp_s, test_size=0.5, stratify=y_temp_s, random_state=42
)

num_classes = len(np.unique(y))
log(f"Number of classes: {num_classes}")

inp_s = Input(shape=(X_train_s.shape[1], X_train_s.shape[2]))
x = LSTM(64)(inp_s)
x = Dropout(0.3)(x)
out_s = Dense(num_classes, activation="softmax")(x)

lstm_model = Model(inp_s, out_s)

lstm_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("lstm_best.h5", save_best_only=True),
    TqdmCallback()
]

log("Training LSTM...")
lstm_model.fit(
    X_train_s, y_train_s,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_s, y_val_s),
    callbacks=callbacks,
    verbose=0
)

lstm_model.save("lstm_final.h5")
log("LSTM saved.")


# ==========================================
# 🔥 STEP 9: XGBOOST
# ==========================================
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1]
}

search = RandomizedSearchCV(
    xgb.XGBClassifier(),
    param_dist,
    n_iter=5,
    cv=3,
    verbose=1,
    n_jobs=-1
)

log("Training XGBoost...")
search.fit(X_train, y_train)

best_xgb = search.best_estimator_
joblib.dump(best_xgb, "xgboost_model.pkl")

log("XGBoost saved.")


# ==========================================
# 📊 STEP 10: EVALUATION
# ==========================================
log("Evaluating XGBoost...")

preds = best_xgb.predict(X_test)

report = classification_report(y_test, preds)
log("\n" + report)

cm = confusion_matrix(y_test, preds)
log(f"Confusion Matrix:\n{cm}")

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


# ==========================================
# 🔥 STEP 11: ROC
# ==========================================
y_bin = label_binarize(y_test, classes=np.unique(y))
probs = best_xgb.predict_proba(X_test)

auc = roc_auc_score(y_bin, probs, multi_class="ovr")
log(f"ROC AUC: {auc}")


# ==========================================
# 🎉 DONE
# ==========================================
log("Training completed successfully.")
print("All models saved. Logs generated.")