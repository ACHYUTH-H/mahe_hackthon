import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "clean_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# 🔹 Helper: Smart Column Detection
# ===========================
def detect_columns(df):
    time_col = None
    value_col = None

    for col in df.columns:
        c = col.lower()

        if "time" in c:
            time_col = col

        elif any(k in c for k in [
            "ratio", "delivery", "throughput", "overhead",
            "bytes", "efficiency", "delay", "packet"
        ]):
            value_col = col

    # fallback (if detection fails)
    if time_col is None:
        time_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return time_col, value_col


# ===========================
# 🔹 1. FIX DELIVERY RATIO
# ===========================
def plot_delivery():
    file = "8_Fragment_Delivery_Ratio.csv"
    df = pd.read_csv(file)

    time_col, value_col = detect_columns(df)

    plt.figure(figsize=(10,5))
    plt.plot(df[time_col], df[value_col])
    plt.xlabel("Simulation Time")
    plt.ylabel("Delivery Ratio")
    plt.title("Packet Delivery Ratio Over Time")
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/delivery_ratio.png")
    plt.close()


# ===========================
# 🔹 2. FIX NETWORK OVERHEAD
# ===========================
def plot_overhead():
    file = "6_Network_Overhead_Bytes.csv"
    df = pd.read_csv(file)

    time_col, value_col = detect_columns(df)

    plt.figure(figsize=(10,5))
    plt.plot(df[time_col], df[value_col])
    plt.xlabel("Simulation Time")
    plt.ylabel("Overhead (Bytes)")
    plt.title("Network Overhead Over Time")
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/network_overhead.png")
    plt.close()


# ===========================
# 🔹 3. FIX EPIDEMIC SUPPRESSION
# ===========================
def plot_epidemic():
    file = "4_Epidemic_Suppression_Log.csv"
    df = pd.read_csv(file)

    time_col, value_col = detect_columns(df)

    plt.figure(figsize=(10,5))
    plt.scatter(df[time_col], df[value_col])
    plt.xlabel("Simulation Time")
    plt.ylabel("Suppression Events")
    plt.title("Epidemic Suppression Events")
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/epidemic_events.png")
    plt.close()


# ===========================
# 🔹 4. FIX POLYKEM EFFICIENCY
# ===========================
def plot_polykem():
    file = "3_PolyKEM_Efficiency_Stats.csv"
    df = pd.read_csv(file)

    time_col, value_col = detect_columns(df)

    plt.figure(figsize=(10,5))
    plt.plot(df[time_col], df[value_col])
    plt.xlabel("Simulation Time")
    plt.ylabel("Efficiency")
    plt.title("PolyKEM Efficiency Over Time")
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/polykem_efficiency.png")
    plt.close()


# ===========================
# 🔹 RUN ALL
# ===========================
if __name__ == "__main__":
    print("Cleaning and generating correct plots...")

    plot_delivery()
    plot_overhead()
    plot_epidemic()
    plot_polykem()

    print(f"✅ Clean plots saved in '{OUTPUT_DIR}'")