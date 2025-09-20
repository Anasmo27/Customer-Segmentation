"""
Customer Segmentation Interactive Script
- Train KMeans on a synthetic dataset (Age, Annual_Income, Spending_Score)
- Let user input one or more customer rows via terminal
- Predict cluster for each input, compute distance to centroid
- Visualize test set clusters + user points and centroids
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import math
import textwrap

# 1) Load dataset from CSV
file_path = r"C:\Users\ASUS\Desktop\python test\Customer_Segmentation\customer_data.csv"  # Update this path as needed
data = pd.read_csv(file_path)

# Ensure required columns exist
required_cols = ["Age", "Annual_Income", "Spending_Score"]
if not all(col in data.columns for col in required_cols):
    raise ValueError(f"CSV file must contain columns: {required_cols}")

# Use only required columns
X = data[required_cols]


# -------------------------
# 2) Train-Test Split
# -------------------------
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# -------------------------
# 3) Preprocessing (Standardize)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 4) Train KMeans on training data
# -------------------------
N_CLUSTERS = 4
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

# compute cluster labels for test set
test_labels = kmeans.predict(X_test_scaled)

# -------------------------
# Helper: parse user input rows
# -------------------------
def parse_user_rows():
    """
    Prompts the user to enter rows. Each row should be:
      Age, Annual_Income, Spending_Score
    Enter empty line to finish.
    Returns list of tuples (Age, Income, Spending)
    """
    print("\nEnter customer rows one per line in this format:")
    print("  Age, Annual_Income, Spending_Score")
    print("Examples:")
    print("  34, 45000, 60")
    print("  23, 120000, 30")
    print("When you finish, press Enter on an empty line.\n")

    rows = []
    while True:
        line = input("Row (or blank to finish): ").strip()
        if line == "":
            break
        try:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                raise ValueError("Expect 3 comma-separated values.")
            age = float(parts[0])
            income = float(parts[1])
            score = float(parts[2])
            rows.append((age, income, score))
        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")
    return rows

# -------------------------
# 5) Get user rows
# -------------------------
user_rows = parse_user_rows()
if len(user_rows) == 0:
    print("No user rows entered — exiting.")
    raise SystemExit

user_df = pd.DataFrame(user_rows, columns=["Age", "Annual_Income", "Spending_Score"])
user_scaled = scaler.transform(user_df.values)

# -------------------------
# 6) Predict clusters & compute distances to centroids
# -------------------------
user_preds = kmeans.predict(user_scaled)
centroids = kmeans.cluster_centers_

# compute Euclidean distance from each user point to its assigned centroid
def euclidean(a, b):
    return math.sqrt(((a - b) ** 2).sum())

distances = [euclidean(user_scaled[i], centroids[user_preds[i]]) for i in range(len(user_preds))]

# Print result table
print("\nResults for entered customers:")
out_df = user_df.copy()
out_df["Predicted_Cluster"] = user_preds
out_df["Distance_to_Centroid"] = np.round(distances, 4)
print(out_df.to_string(index=False))

# -------------------------
# 7) Silhouette (optional)
# -------------------------
# We can compute silhouette score for test set alone, and for combined test+user if possible
try:
    if len(np.unique(test_labels)) > 1:
        sil_test = silhouette_score(X_test_scaled, test_labels)
    else:
        sil_test = float("nan")
except Exception:
    sil_test = float("nan")

# combined
combined_X = np.vstack([X_test_scaled, user_scaled])
combined_labels = np.concatenate([test_labels, user_preds])

try:
    sil_comb = silhouette_score(combined_X, combined_labels) if len(np.unique(combined_labels)) > 1 else float("nan")
except Exception:
    sil_comb = float("nan")

print(f"\nSilhouette Score (test only)    : {sil_test:.4f}")
print(f"Silhouette Score (test + input) : {sil_comb:.4f}")

# -------------------------
# 8) Visualization
# -------------------------
# We'll visualize using Annual_Income (x) vs Spending_Score (y).
# Age is encoded as marker size (optional).
fig, ax = plt.subplots(figsize=(10, 6))

# plot test set points colored by cluster
scatter = ax.scatter(
    X_test["Annual_Income"],
    X_test["Spending_Score"],
    c=test_labels,
    cmap="tab10",
    alpha=0.6,
    edgecolor="k",
    s=50,
    label="Test points"
)

# plot cluster centroids (transform centroids back to original scale for plotting)
centroids_orig = scaler.inverse_transform(centroids)
ax.scatter(
    centroids_orig[:, 1],  # Annual_Income
    centroids_orig[:, 2],  # Spending_Score
    marker="X", s=200, c="black", label="Centroids"
)

# plot user points with special markers and annotate
for i, row in user_df.iterrows():
    income = row["Annual_Income"]
    score = row["Spending_Score"]
    cluster_i = user_preds[i]
    dist_i = distances[i]
    ax.scatter(income, score, marker="D", s=120, edgecolor="white",
               label=f"User {i+1} (Cluster {cluster_i})" if i == 0 else None,
               c=[cluster_i], cmap="tab10")
    ax.annotate(f"C{cluster_i}, d={dist_i:.2f}",
                (income, score),
                textcoords="offset points", xytext=(6,6), fontsize=9, weight="bold")

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation (Test set + Your Inputs)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
plt.tight_layout()
plt.show()

# -------------------------
# End
# -------------------------
print("\nDone. You can rerun the script to try different inputs or change N_CLUSTERS.")
