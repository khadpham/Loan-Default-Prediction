import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('../../data/raw/train.csv')

# Display the first few rows and get a general description
df.info()
df.head()

df_encoded=df.drop(['Loan Title',"Accounts Delinquent",'Batch Enrolled','Sub Grade','Payment Plan','ID'],axis=1)


# Final correlation analysis
correlation_matrix_final = df_encoded.select_dtypes(include=np.number).corr()

correlation_matrix_final.to_csv("../../data/interim/001_cleaned.csv")

