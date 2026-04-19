import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("8_Fragment_Delivery_Ratio.csv")

plt.plot(df.iloc[:,0], df.iloc[:,1])
plt.xlabel("Time")
plt.ylabel("Delivery Ratio")
plt.title("Fragment Delivery Ratio Over Time")
plt.grid()

plt.savefig("delivery_ratio.png")
plt.show()