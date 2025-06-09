import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

categories = ["ACS", "DynPhe", "MultiCol", "Hybrid"]
sizes = ["1k", "10k", "50k"]

all_data = []

for category in categories:
    for size in sizes:
        filename = f"{category}_{size}_aco_dna_results.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # convert size labels
            if size == "1k":
                size_label = "1.000 bases"
            elif size == "10k":
                size_label = "10.000 bases"
            else:  # 50k
                size_label = "50.000 bases"

            df["Category"] = category
            df["Size"] = size_label
            all_data.append(df)
        else:
            print(f"File not found: {filename}")

df_all = pd.concat(all_data, ignore_index=True)

# set order for plotting
size_order = ["1.000 bases", "10.000 bases", "50.000 bases"]
df_all["Size"] = pd.Categorical(df_all["Size"], categories=size_order, ordered=True)

plt.figure(figsize=(10, 6))          # length difference
sns.boxplot(data=df_all, x="Size", y="optimization_time", hue="Category", palette="Set2", width=0.5)

plt.title("Optimization Time by Algorithm and Input Size")
plt.ylabel("Optimization Time (seconds)")
plt.xlabel("Input Size")
plt.legend(title="Algorithm")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
