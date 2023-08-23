from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("../../data/interim/002_final.csv")
df.fillna(df.median(), inplace=True)
df.info()

X = df.drop(columns=["Loan Status"])
y = df["Loan Status"]

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_classifier.fit(X, y)

# Get the feature importances
feature_importances = rf_classifier.feature_importances_

# Creating a DataFrame for feature importances
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sorting the features based on importance
sorted_features_df = features_df.sort_values(by='Importance', ascending=False).head(15)

selected_features = sorted_features_df['Feature'].tolist()
df_selected = df[selected_features + ['Loan Status']]

df_selected.to_csv("../../data/interim/003_modelling.csv", index=False)
