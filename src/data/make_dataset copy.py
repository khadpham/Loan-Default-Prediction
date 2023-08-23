import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../../data/raw/train.csv')

# Display the first few rows and get a general description
df.info()

df.head()
numeric_columns = ['Loan Amount', 'Funded Amount', 'Funded Amount Investor', 'Term', 'Interest Rate', 'Debit to Income', 'Delinquency - two years', 'Inquires - six months', 'Open Account', 'Public Record', 'Revolving Balance', 'Revolving Utilities', 'Total Accounts', 'Total Received Interest', 'Total Received Late Fee', 'Recoveries', 'Collection Recovery Fee', 'Collection 12 months Medical', 'Accounts Delinquent', 'Total Collection Amount', 'Total Current Balance', 'Total Revolving Credit Limit']

# Outlier detection using Isolation Forest
isolation_forest = IsolationForest(contamination=0.05)
outliers_isolation_forest = isolation_forest.fit_predict(df[numeric_columns])

# Outlier detection using LOF
lof = LocalOutlierFactor(contamination=0.05)
outliers_lof = lof.fit_predict(df[numeric_columns])

# Outlier detection using Robust Z-score
median = np.median(df[numeric_columns], axis=0)
mad = np.median(np.abs(df[numeric_columns] - median), axis=0)
robust_z_scores = 0.6745 * (df[numeric_columns] - median) / mad
outliers_robust_z = np.any(np.abs(robust_z_scores) > 3, axis=1)

# Count the number of outliers
num_outliers_isolation_forest = np.sum(outliers_isolation_forest == -1)
num_outliers_lof = np.sum(outliers_lof == -1)
num_outliers_robust_z = np.sum(outliers_robust_z == True)

# Create subplots for the three graphs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Isolation Forest
axes[0].scatter(df['Loan Amount'], df['Interest Rate'], c=outliers_isolation_forest, label='Isolation Forest')
axes[0].set_xlabel('Loan Amount')
axes[0].set_ylabel('Interest Rate')
axes[0].set_title('Isolation Forest')

# Plot LOF
axes[1].scatter(df['Loan Amount'], df['Interest Rate'], c=outliers_lof, label='LOF')
axes[1].set_xlabel('Loan Amount')
axes[1].set_ylabel('Interest Rate')
axes[1].set_title('LOF')

# Plot Robust Z-score
axes[2].scatter(df['Loan Amount'], df['Interest Rate'], c=outliers_robust_z, label='Robust Z-score')
axes[2].set_xlabel('Loan Amount')
axes[2].set_ylabel('Interest Rate')
axes[2].set_title('Robust Z-score')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()

# Print the number of outliers detected
print("Number of outliers detected by Isolation Forest:", num_outliers_isolation_forest)
print("Number of outliers detected by LOF:", num_outliers_lof)
print("Number of outliers detected by Robust Z-score:", num_outliers_robust_z)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, column in enumerate(numeric_columns):
    axes[i].scatter(df[column], df['Interest Rate'], c=outliers_isolation_forest, label='Outliers')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Interest Rate')
    axes[i].set_title(f'Outliers Detected in {column}')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()

df_filtered = df[outliers_isolation_forest != -1]

plt.hist(df_filtered['Loan Amount'], bins=20)
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Loan Amount (Filtered Data)')
plt.show()

# Example: Calculate the mean and standard deviation of the Interest Rate
mean_interest_rate = df_filtered['Interest Rate'].mean()
std_interest_rate = df_filtered['Interest Rate'].std()
print("Mean Interest Rate:", mean_interest_rate)
print("Standard Deviation of Interest Rate:", std_interest_rate)
