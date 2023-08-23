# Data Exploration and Preprocessing Report

## Overview

This report summarizes the steps taken for data exploration and preprocessing on a given dataset. The process aims to ensure data quality and relevance for subsequent analyses or modeling.

## Table of Contents

- [Initial Data Exploration](#initial-data-exploration)
- [Outlier Detection](#outlier-detection)
- [Feature Removal](#feature-removal)
- [Correlation Analysis](#correlation-analysis)

## Initial Data Exploration

The dataset was loaded into a pandas dataframe, and the initial exploration was conducted. This included:
- Checking the data's shape and size.
- Viewing the initial rows for a preliminary understanding.
- Computing descriptive statistics for numerical variables.
- Computing frequency counts for categorical variables.

## Outlier Detection

Three methods were initially considered for outlier detection: Z-score, Isolation Forest, and Local Outlier Factor (LOF). Due to memory constraints, only the Z-score and Isolation Forest methods were used.

**Findings**:
- **Z-score**: Detected a significant number of outliers.
- **Isolation Forest**: Detected 3,374 outliers. This method was chosen for outlier removal.

## Feature Removal

Several columns were removed due to being deemed unimportant or redundant:
- **ID**: A unique identifier for each entry.
- **Batch Enrolled**: Removed upon request.
- **Accounts Delinquent**: Removed upon request.
- **Payment Plan**: Had a single unique value and was removed.

## Correlation Analysis

A correlation matrix was computed, and a heatmap was generated to visualize the relationships between the numerical variables. This helped in understanding potential multicollinearity and the strength of relationships between different variables.

**Note**: This report serves as a documentation of the data exploration and preprocessing steps. Additional steps and modifications might be required based on specific analyses or modeling objectives.

# Detailed Data Exploration and Preprocessing Report

## Overview

This report provides an in-depth view of the data exploration and preprocessing steps undertaken on the dataset. The dataset underwent several cleaning, preprocessing, and exploration steps to ensure its readiness for modeling or further analysis.

## Table of Contents

- [Data Cleaning](#data-cleaning)
- [Descriptive Statistics & Data Visualization](#descriptive-statistics--data-visualization)
- [Correlation Analysis](#correlation-analysis)
- [Multivariate Analysis](#multivariate-analysis)

## Data Cleaning

The initial dataset was inspected for duplicates and constant columns. The following steps were taken:

1. **Duplicates**: Checked for and removed any duplicate rows.
2. **Constant Columns**: Removed columns with a single unique value to reduce dimensionality.

## Descriptive Statistics & Data Visualization

The dataset's characteristics and distributions were explored using various charts:

1. **Loan Amount**: A histogram was plotted to observe the distribution. Most loans were found to be in the range of $5,000 to $20,000.
2. **Interest Rate**: A histogram showcased the majority of loans having an interest rate between 9% and 15%.
3. **Grade**: A bar plot depicted the count of loans by grade, revealing a skewed distribution with certain grades being more frequent.
4. **Loan Amount by Grade**: A box plot showed the spread of loan amounts across different grades.

## Correlation Analysis

A heatmap was used to visualize correlations:

- **Strong Positive Correlations**: Variables like "Funded Amount" and "Loan Amount" exhibited strong positive correlations.
- **Other Correlations**: Some other variables also showed significant correlations, providing insights into potential multicollinearity.

## Multivariate Analysis

Relationships between multiple variables were explored:

- A scatter plot of "Loan Amount" vs. "Interest Rate" was drawn, with colors indicating different "Grades". This visual helped discern patterns and relationships between the three variables.

---

**Note**: This report documents the data exploration and preprocessing steps in detail. The cleaned dataset is now ready for further analyses, modelling, or other data-driven tasks.
---


# Comprehensive Data Preprocessing and Transformation Report

## Overview

This report delves into the various preprocessing and transformation steps executed on the dataset. It aims to document the methodologies and results, ensuring the data's readiness for further analysis or modeling.

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
    - [Formatting](#formatting)
    - [Encoding](#encoding)
    - [Imputing](#imputing)
    - [Sampling](#sampling)
    - [Splitting](#splitting)
- [Data Transformation](#data-transformation)
    - [Scaling](#scaling)
    - [Normalizing](#normalizing)
    - [Discretizing](#discretizing)
    - [Combining](#combining)

## Data Preprocessing

### Formatting

Initial inspection of the dataset ensured consistent formatting. Specifically:
- 'Grade' and 'Sub Grade' columns were consistently formatted.
- 'Loan Title' had variations representing similar information (e.g., "Debt Consolidation" vs. "Debt consolidation").

### Encoding

Categorical variables were transformed to numerical formats:
- One-hot encoding was applied to columns with few unique values, generating binary columns.
- Ordinal encoding was used for 'Grade' and 'Sub Grade' due to their inherent order.

### Imputing

No missing values were detected in the dataset, negating the need for imputation.

### Sampling

The full dataset was retained for comprehensive exploration without sampling.

### Splitting

Data splitting into training, validation, and test sets will be considered during modeling phases.

## Data Transformation

### Scaling

Numerical features were previously scaled using the StandardScaler, resulting in a mean of 0 and a standard deviation of 1 for these features.

### Normalizing

Logarithmic transformations were applied to certain columns exhibiting positive skewness to achieve more symmetric distributions.

### Discretizing

The 'Loan Amount' column was discretized into three categories: 'Low', 'Medium', and 'High', dividing the dataset approximately equally.

### Combining

No combining methods like PCA were applied at this stage.

---

This report provides a clear view of the preprocessing and transformation techniques applied to the dataset. It serves as documentation and reference for understanding the current state of the dataset.
