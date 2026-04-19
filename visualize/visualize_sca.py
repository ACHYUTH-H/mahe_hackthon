import pandas as pd
import matplotlib.pyplot as plt

data = []

with open("General-#0.sca") as f:
    for line in f:
        if line.startswith("scalar"):
            parts = line.split()
            metric = parts[2]
            value = float(parts[3])
            data.append([metric, value])

df = pd.DataFrame(data, columns=["Metric", "Value"])

# Plot top metrics
df.groupby("Metric").mean().plot(kind="bar", figsize=(10,5))
plt.title("Network Metrics Summary")
plt.tight_layout()
plt.savefig("sca_plot.png")
plt.show()