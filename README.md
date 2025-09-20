Customer Segmentation with K-Means

This project demonstrates customer segmentation using the K-Means clustering algorithm. It trains a clustering model on customer data (age, annual income, and spending score), then allows users to input their own customer data for cluster prediction, analysis, and visualization.

🚀 Features

Load customer dataset (CSV format).

Preprocess data using StandardScaler.

Train K-Means clustering model with configurable number of clusters.

Interactive user input:

Enter customer details (Age, Annual Income, Spending Score).

Predict cluster assignment for each customer.

Compute Euclidean distance from each customer to the assigned centroid.

Evaluation:

Compute Silhouette Score for test set and combined data.

Visualization:

Plot test set clusters with centroids.

Highlight and annotate user-provided customers.

📂 Project Structure
Customer_Segmentation/
│── Customer_Segmentation.py   # Main script
│── customer_data.csv          # Input dataset (sample data required)
│── README.md                  # Project documentation

📊 Dataset

The dataset must include the following columns:

Age → Age of customer

Annual_Income → Yearly income of customer

Spending_Score → Customer spending score (1–100)

👉 Example (customer_data.csv):

Age,Annual_Income,Spending_Score
19,15000,39
35,60000,81
26,45000,6
45,80000,77

⚙️ Installation

Clone this repository or download the script.

Install required dependencies:

pip install numpy pandas matplotlib scikit-learn


Place your dataset in the project directory as customer_data.csv.

Update the dataset path in the script if needed:

file_path = r"C:\path\to\customer_data.csv"

▶️ Usage

Run the script:

python Customer_Segmentation.py

Example Input (interactive):
Enter customer rows one per line in this format:
  Age, Annual_Income, Spending_Score
Examples:
  34, 45000, 60
  23, 120000, 30
When you finish, press Enter on an empty line.

Row (or blank to finish): 28, 52000, 70
Row (or blank to finish): 45, 88000, 30
Row (or blank to finish):

Example Output (table):
 Results for entered customers:
 Age  Annual_Income  Spending_Score  Predicted_Cluster  Distance_to_Centroid
  28          52000              70                  2                 0.2154
  45          88000              30                  1                 0.3189

Visualization Example:

Scatter plot of test data points colored by cluster.

Centroids marked with black X.

User inputs shown with diamond markers + annotations.

📈 Evaluation

Silhouette Score (test only) → evaluates clustering quality on unseen data.

Silhouette Score (test + input) → evaluates clustering including user-provided rows.

🔧 Customization

Change number of clusters:

N_CLUSTERS = 4


Adjust visualization (marker size, colors, etc.).

Extend dataset with more features (e.g., gender, region).

📌 Requirements

Python 3.8+

NumPy

Pandas

Matplotlib

Scikit-learn

📜 License

This project is open-source and available for educational purposes.
