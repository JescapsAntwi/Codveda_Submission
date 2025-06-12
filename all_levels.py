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
from sklearn.datasets import load_iris, make_classification, make_blobs
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

# One-hot encoding for city (no ordinal relationship)
city_dummies = pd.get_dummies(df['city'], prefix='city')
df = pd.concat([df, city_dummies], axis=1)
print("   - Applied one-hot encoding to city (nominal)")

# Step 3: Normalize/Standardize numerical features
print("\n3. Normalizing Numerical Features:")

# Select numerical columns for scaling
numerical_cols = ['age', 'income', 'experience']
scaler = StandardScaler()

# Fit and transform numerical features
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("   - Applied StandardScaler to numerical features")
print(f"   - Scaled columns: {numerical_cols}")

# Step 4: Prepare final dataset and split
print("\n4. Preparing Final Dataset:")

# Select features for modeling (excluding original categorical columns)
feature_cols = ['age', 'income', 'education_encoded', 'experience'] + list(city_dummies.columns)
X = df[feature_cols]
y = df['target']

print(f"   - Final feature set shape: {X.shape}")
print(f"   - Target variable shape: {y.shape}")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   - Training set: {X_train.shape[0]} samples")
print(f"   - Testing set: {X_test.shape[0]} samples")
print("   - Preprocessing completed successfully!")


print("\n--- TASK 2: LINEAR REGRESSION MODEL BUILDING ---")

from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()

X_housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
y_housing = housing_data.target

print(f"Housing dataset shape: {X_housing.shape}")
print(f"Features: {list(X_housing.columns)}")

# Preprocess the housing data
print("\n1. Preprocessing Housing Data:")
housing_scaler = StandardScaler()
X_housing_scaled = housing_scaler.fit_transform(X_housing)
print("   - Applied StandardScaler to all features")

# Split the data
X_train_house, X_test_house, y_train_house, y_test_house = train_test_split(
    X_housing_scaled, y_housing, test_size=0.2, random_state=42
)

print(f"   - Training samples: {X_train_house.shape[0]}")
print(f"   - Testing samples: {X_test_house.shape[0]}")

# Train linear regression model
print("\n2. Training Linear Regression Model:")
lr_model = LinearRegression()
lr_model.fit(X_train_house, y_train_house)
print("   - Model training completed")

# Make predictions
y_pred_house = lr_model.predict(X_test_house)

# Evaluate the model
print("\n3. Model Evaluation:")
r2 = r2_score(y_test_house, y_pred_house)
mse = mean_squared_error(y_test_house, y_pred_house)
rmse = np.sqrt(mse)

print(f"   - R-squared Score: {r2:.4f}")
print(f"   - Mean Squared Error: {mse:.4f}")
print(f"   - Root Mean Squared Error: {rmse:.4f}")

# Interpret model coefficients
print("\n4. Model Coefficients Interpretation:")
feature_importance = pd.DataFrame({
    'Feature': housing_data.feature_names,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("   Top 5 most important features:")
for i, row in feature_importance.head().iterrows():
    print(f"   - {row['Feature']}: {row['Coefficient']:.4f}")

print("\n--- TASK 3: K-NEAREST NEIGHBORS CLASSIFIER ---")

# Load Iris dataset for classification
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Iris dataset shape: {X_iris.shape}")
print(f"Classes: {iris.target_names}")

# Split the data
print("\n1. Preparing Iris Dataset:")
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

# Scale the features
iris_scaler = StandardScaler()
X_train_iris_scaled = iris_scaler.fit_transform(X_train_iris)
X_test_iris_scaled = iris_scaler.transform(X_test_iris)

print(f"   - Training samples: {X_train_iris_scaled.shape[0]}")
print(f"   - Testing samples: {X_test_iris_scaled.shape[0]}")

# Test different values of K
print("\n2. Testing Different K Values:")
k_values = [1, 3, 5, 7, 9, 11]
k_results = []

for k in k_values:
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_iris_scaled, y_train_iris)
    
    # Make predictions
    y_pred_knn = knn.predict(X_test_iris_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_iris, y_pred_knn)
    k_results.append(accuracy)
    
    print(f"   - K={k}: Accuracy = {accuracy:.4f}")

# Find best K value
best_k = k_values[np.argmax(k_results)]
print(f"\n   Best K value: {best_k} with accuracy: {max(k_results):.4f}")

# Train final KNN model with best K
print("\n3. Final KNN Model Evaluation:")
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_iris_scaled, y_train_iris)
y_pred_final = final_knn.predict(X_test_iris_scaled)

# Detailed evaluation
accuracy = accuracy_score(y_test_iris, y_pred_final)
conf_matrix = confusion_matrix(y_test_iris, y_pred_final)
class_report = classification_report(y_test_iris, y_pred_final, target_names=iris.target_names)

print(f"   - Final Accuracy: {accuracy:.4f}")
print(f"   - Confusion Matrix:\n{conf_matrix}")
print(f"   - Classification Report:\n{class_report}")

print("\n" + "=" * 10 + " LOGISTIC REGRESSION MODEL IMPLEMENTATION FOR BINARY CLASSIFICATION " + "=" * 10)

# Create a binary classification dataset
print("1. Creating Binary Classification Dataset:")
X_binary, y_binary = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=3,
    n_classes=2, random_state=42
)

# Add feature names
feature_names = [f'feature_{i+1}' for i in range(X_binary.shape[1])]
X_binary_df = pd.DataFrame(X_binary, columns=feature_names)

print(f"   - Dataset shape: {X_binary.shape}")
print(f"   - Class distribution: {np.bincount(y_binary)}")

# Split and scale the data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

binary_scaler = StandardScaler()
X_train_bin_scaled = binary_scaler.fit_transform(X_train_bin)
X_test_bin_scaled = binary_scaler.transform(X_test_bin)

# Train logistic regression model
print("\n2. Training Logistic Regression Model:")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_bin_scaled, y_train_bin)

# Make predictions
y_pred_log = log_reg.predict(X_test_bin_scaled)
y_pred_proba = log_reg.predict_proba(X_test_bin_scaled)[:, 1]

# Evaluate the model
print("\n3. Model Evaluation:")
accuracy = accuracy_score(y_test_bin, y_pred_log)
conf_matrix = confusion_matrix(y_test_bin, y_pred_log)

print(f"   - Accuracy: {accuracy:.4f}")
print(f"   - Confusion Matrix:\n{conf_matrix}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"   - ROC AUC Score: {roc_auc:.4f}")

# Interpret coefficients and odds ratios
print("\n4. Coefficient Interpretation:")
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': log_reg.coef_[0],
    'Odds_Ratio': np.exp(log_reg.coef_[0])
}).sort_values('Coefficient', key=abs, ascending=False)

print("   Top 5 most important features:")
for i, row in coefficients.head().iterrows():
    print(f"   - {row['Feature']}: Coef={row['Coefficient']:.4f}, OR={row['Odds_Ratio']:.4f}")

print("\n============= DECISION TREES ================")

# Use Iris dataset for decision tree
print("1. Training Decision Tree on Iris Dataset:")

# Train decision tree
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_classifier.fit(X_train_iris, y_train_iris)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test_iris)

# Evaluate the model
accuracy_dt = accuracy_score(y_test_iris, y_pred_dt)
f1_dt = f1_score(y_test_iris, y_pred_dt, average='weighted')
conf_matrix_dt = confusion_matrix(y_test_iris, y_pred_dt)

print(f"   - Accuracy: {accuracy_dt:.4f}")
print(f"   - F1-Score: {f1_dt:.4f}")
print(f"   - Confusion Matrix:\n{conf_matrix_dt}")

# Feature importance
print("\n2. Feature Importance Analysis:")
feature_importance_dt = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': dt_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("   Feature importance ranking:")
for i, row in feature_importance_dt.iterrows():
    print(f"   - {row['Feature']}: {row['Importance']:.4f}")

# Pruning demonstration
print("\n3. Tree Pruning Analysis:")
# Train trees with different max_depth values
depths = [2, 3, 4, 5, 6, None]
pruning_results = []

for depth in depths:
    dt_temp = DecisionTreeClassifier(random_state=42, max_depth=depth)
    dt_temp.fit(X_train_iris, y_train_iris)
    y_pred_temp = dt_temp.predict(X_test_iris)
    acc_temp = accuracy_score(y_test_iris, y_pred_temp)
    pruning_results.append(acc_temp)
    
    depth_str = str(depth) if depth is not None else "None"
    print(f"   - Max depth {depth_str}: Accuracy = {acc_temp:.4f}")

best_depth = depths[np.argmax(pruning_results)]
print(f"   - Best max_depth: {best_depth}")

print("============== K MEANS CLUSTERING IMPLEMENTATION ================")
# Create clustering dataset
print("1. Creating Clustering Dataset:")
X_cluster, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

print(f"   - Dataset shape: {X_cluster.shape}")

# Scale the data
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using elbow method
print("\n2. Finding Optimal Number of Clusters (Elbow Method):")
k_range = range(1, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    print(f"   - K={k}: Inertia = {kmeans.inertia_:.2f}")

# Find elbow point (simplified method)
# Calculate the rate of decrease in inertia
rate_decrease = []
for i in range(1, len(inertias)):
    rate_decrease.append(inertias[i-1] - inertias[i])

# Find the point where rate of decrease starts to slow down significantly
optimal_k = rate_decrease.index(max(rate_decrease[:5])) + 2  # +2 because we start from k=2
print(f"   - Suggested optimal K: {optimal_k}")

# Apply K-Means with optimal K
print(f"\n3. Applying K-Means with K={optimal_k}:")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_cluster_scaled)

# Analyze clusters
print("   - Clustering completed")
print(f"   - Cluster centers shape: {final_kmeans.cluster_centers_.shape}")

# Cluster analysis
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
print("   - Cluster sizes:")
for label, count in zip(unique_labels, counts):
    print(f"     Cluster {label}: {count} points")

# Calculate silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
print(f"   - Average Silhouette Score: {silhouette_avg:.4f}")

print("========== RANDOM FOREST CLASSIFIER IMPLEMENTATION ===========")
print("1. Creating Complex Classification Dataset:")
X_complex, y_complex = make_classification(
    n_samples=1500, n_features=20, n_informative=15, n_redundant=3,
    n_classes=3, random_state=42
)

complex_feature_names = [f'feature_{i+1}' for i in range(X_complex.shape[1])]
print(f"   - Dataset shape: {X_complex.shape}")
print(f"   - Classes: {np.unique(y_complex)}")
print(f"   - Class distribution: {np.bincount(y_complex)}")

# Split and scale the data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_complex, y_complex, test_size=0.2, random_state=42, stratify=y_complex
)

rf_scaler = StandardScaler()
X_train_rf_scaled = rf_scaler.fit_transform(X_train_rf)
X_test_rf_scaled = rf_scaler.transform(X_test_rf)

# Hyperparameter tuning
print("\n2. Hyperparameter Tuning:")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid search with cross-validation
rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_rf_scaled, y_train_rf)

print(f"   - Best parameters: {grid_search.best_params_}")
print(f"   - Best CV score: {grid_search.best_score_:.4f}")