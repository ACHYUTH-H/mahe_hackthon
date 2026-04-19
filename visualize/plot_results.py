import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Plot your original Fragment Delivery CSV
# ==========================================
def plot_delivery_ratio():
    try:
        df = pd.read_csv('8_Fragment_Delivery_Ratio.csv')
        plt.figure(figsize=(10, 6))
        
        # Assuming column 0 is X (e.g., Time/Node) and column 1 is Y (Delivery Ratio)
        x_col, y_col = df.columns[0], df.columns[1]
        
        plt.plot(df[x_col], df[y_col], marker='o', color='#1f77b4', linewidth=2)
        plt.title('Fragment Delivery Ratio', fontsize=14, fontweight='bold')
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('1_Delivery_Ratio.png', dpi=300)
        print("Saved: 1_Delivery_Ratio.png")
    except Exception as e:
        print(f"Skipping Delivery Ratio: {e}")

# ==========================================
# 2. Plot OMNeT++ Vectors (Line Charts)
# ==========================================
def plot_vectors():
    try:
        # scavetool often exports wide or long formats. 
        df = pd.read_csv('vectors.csv')
        plt.figure(figsize=(10, 6))
        
        # Usually, scavetool outputs 'vectime' and 'vecvalue' columns
        if 'vectime' in df.columns and 'vecvalue' in df.columns:
            # Plot the first vector found as an example
            plt.plot(df['vectime'], df['vecvalue'], color='#ff7f0e')
        else:
            # Fallback: plot the first two columns
            plt.plot(df.iloc[:, 0], df.iloc[:, 1], color='#ff7f0e')

        plt.title('Network Vector Data (Over Time)', fontsize=14, fontweight='bold')
        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('2_Vector_Graph.png', dpi=300)
        print("Saved: 2_Vector_Graph.png")
    except Exception as e:
        print(f"Skipping Vectors: {e}")

# ==========================================
# 3. Plot OMNeT++ Scalars (Bar Charts)
# ==========================================
def plot_scalars():
    try:
        df = pd.read_csv('scalars.csv')
        plt.figure(figsize=(10, 6))
        
        # Filter for actual scalar values (scavetool CSVs have specific columns)
        # We will plot the first 10 modules for a clean chart
        df_scalars = df[df['type'] == 'scalar'].head(10) 
        
        plt.bar(df_scalars['module'] + '\n' + df_scalars['name'], df_scalars['value'], color='#2ca02c')
        plt.title('Network Scalar Summaries', fontsize=14, fontweight='bold')
        plt.xlabel('Module & Metric', fontsize=12)
        plt.ylabel('Recorded Value', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('3_Scalar_Chart.png', dpi=300)
        print("Saved: 3_Scalar_Chart.png")
    except Exception as e:
        print(f"Skipping Scalars: {e}")

# Run all functions
if __name__ == '__main__':
    print("Generating presentation charts...")
    plot_delivery_ratio()
    plot_vectors()
    plot_scalars()
    print("Done! Check your folder for the PNG files.")
    plt.show() # This will pop up the windows so you can see them immediately