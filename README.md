



# 🚗 V2X Attack Detection System (Hierarchical ML Framework)

This project implements a **vehicle-level + RSU-level hierarchical attack detection system** for Vehicular Named Data Networking (VNDN).  
The pipeline processes raw simulation data, extracts features, trains models, and performs real-time inference.

---

## 📁 Project Structure
├── models/
├── visualize/
├── confusion_matrix.png
├── feature_engineering3.py
├── final_rsu_model.pkl
├── merge.py
├── model_training4.py
├── requirements.txt


---

## ⚙️ Installation

```bash
pip install -r requirements.txt

Dataset

Download the dataset from:

https://ieee-dataport.org/documents/veremivndn-misbehavior-vehicular-named-data-network



Pipeline Workflow
1️⃣ Merge Dataset

Combine raw simulation outputs into a single dataset.

python merge.py

Output:

Merged dataset (CSV format)

2️⃣ Feature Engineering

Extract meaningful features required for attack detection.

python feature_engineering3.py

Generated Features:

interest_rate
cache_hit / cache_miss
delay_mean
drop_rate
burst_ratio
interest_to_delay_ratio


3️⃣ Model Training

Train the machine learning models.

python model_training4.py

Outputs:

Vehicle-level model
RSU-level hierarchical model (final_rsu_model.pkl)
Evaluation metrics
Confusion matrix


# We have implemented the entire project using omnet, sumo and veins! and we cant add this in the github as the size of the files are very high
