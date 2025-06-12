#load all necessary libraries for this internship assignment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           mean_squared_error, r2_score, roc_curve, auc, f1_score)
#from sklearn.datasets import load_iris, load_boston, make_classification, make_blobs
from sklearn.datasets import fetch_california_housing, fetch_openml
import warnings
warnings.filterwarnings('ignore')

print("\n"+ "=" * 50)
print("LEVEL 1 - BASIC TASKS")
print("=" * 50)

print("\n--- TASK 1: DATA PREPROCESSING ---")

# Create a sample dataset with missing values and categorical variables
np.random.seed(42)
n_samples = 1000

# Generate sample data
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
    'experience': np.random.randint(0, 30, n_samples),
    'target': np.random.randint(0, 2, n_samples)  # Binary target
}

# Create DataFrame
df = pd.DataFrame(data)

# Introduce missing values randomly
missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[missing_indices[:50], 'income'] = np.nan
df.loc[missing_indices[50:], 'experience'] = np.nan

print(f"Original dataset shape: {df.shape}")
print(f"Missing values per column:\n{df.isnull().sum()}")


# Step 1: Handle missing data
print("\n1. Handling Missing Data:")

# Fill missing income with median (robust to outliers)
median_income = df['income'].median()
df['income'].fillna(median_income, inplace=True)
print(f"   - Filled missing income values with median: {median_income:.2f}")

# Fill missing experience with mean
mean_experience = df['experience'].mean()
df['experience'].fillna(mean_experience, inplace=True)
print(f"   - Filled missing experience values with mean: {mean_experience:.2f}")

# Label encoding for education (ordinal relationship)
education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
df['education_encoded'] = df['education'].map(education_mapping)
print("   - Applied label encoding to education (ordinal)")