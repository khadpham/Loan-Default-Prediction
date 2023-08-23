# 1. Further Data Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check for duplicate rows
duplicate_rows = df_cleaned_final[df_cleaned_final.duplicated()]

# If duplicates are found, remove them
if not duplicate_rows.empty:
    df_cleaned_final = df_cleaned_final.drop_duplicates()

# Check for constant columns (columns with a single unique value)
constant_columns = [col for col in df_cleaned_final.columns if df_cleaned_final[col].nunique() == 1]

# If constant columns are found, remove them
if constant_columns:
    df_cleaned_final = df_cleaned_final.drop(columns=constant_columns)

df_cleaned_final.shape, duplicate_rows.shape, constant_columns

# 2. Descriptive Statistics & Data Visualization

# Descriptive statistics for numerical columns
desc_stats = df_cleaned_final.describe()

# Visualization of the distribution of some key numerical columns
plt.figure(figsize=(15, 10))

# Histogram for "Loan Amount"
plt.subplot(2, 2, 1)
sns.histplot(df_cleaned_final['Loan Amount'], kde=True, bins=30)
plt.title('Distribution of Loan Amount')

# Histogram for "Interest Rate"
plt.subplot(2, 2, 2)
sns.histplot(df_cleaned_final['Interest Rate'], kde=True, bins=30)
plt.title('Distribution of Interest Rate')

# Bar plot for "Grade"
plt.subplot(2, 2, 3)
sns.countplot(data=df_cleaned_final, x='Grade', order=df_cleaned_final['Grade'].value_counts().index)
plt.title('Count of Loans by Grade')

# Box plot for "Loan Amount" by "Grade"
plt.subplot(2, 2, 4)
sns.boxplot(data=df_cleaned_final, x='Grade', y='Loan Amount')
plt.title('Loan Amount by Grade')

plt.tight_layout()
plt.show()

desc_stats

# 3. Correlation Analysis

# Computing the correlation matrix for the cleaned data
correlation_matrix = df_cleaned_final.select_dtypes(include=np.number).corr()

# Visualizing the correlation matrix using a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1, annot=False, linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# 4. Multivariate Analysis

# Scatter plot of "Loan Amount" vs "Interest Rate" with hue based on "Grade"
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_cleaned_final, x="Loan Amount", y="Interest Rate", hue="Grade", palette="rainbow", alpha=0.6)
plt.title('Loan Amount vs Interest Rate by Grade')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()

# Checking for unique data types in the dataset
data_types = df_cleaned_final.dtypes

# Extracting columns that are of type 'object' to check for potential inconsistencies
object_columns = data_types[data_types == 'object'].index

# Displaying unique values for each object column to identify any inconsistencies
unique_values_in_object_columns = {col: df_cleaned_final[col].unique() for col in object_columns}

unique_values_in_object_columns

df_cleaned_final.to_csv("../../data/interim/001_cleaned.csv", index=False)
df_cleaned_final=pd.read_csv("../../data/interim/001_cleaned.csv")

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df_cleaned_final=pd.get_dummies(df_encoded,columns=['Term', 'Grade', 'Employment Duration', 'Verification Status',
       'Initial List Status', 'Application Type'],drop_first=True)

df_cleaned_final.to_csv("../../data/interim/003_rough.csv", index=False)