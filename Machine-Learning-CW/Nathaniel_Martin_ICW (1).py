# -------------------------------------------------------
# Machine Learning Coursework - CHD Classification
# Dataset: Heart Disease Risk Factors
# Task: EDA, Ridge Logistic Regression, Classifier Comparison
# -------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report

# Load dataset
df = pd.read_csv("heart-disease.csv")

# Display basic dataset information
df.info()  # Shows data types, non-null counts

# Summary statistics for numerical features
df.describe()

# Histograms of numerical features (to assess distribution and skewness)
numerical_features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'age', 'alcohol']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], bins=20, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Check for missing values in the dataset
print(df.isnull().sum())

# Encode categorical variable 'famhist' to numeric values
df['famhist'] = df['famhist'].map({'Present': 1, 'Absent': 0})

# Visualize correlation between features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Identify potential outliers using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['sbp', 'tobacco', 'ldl', 'adiposity', 'age']])
plt.title("Outlier Detection in Features")
plt.xticks(rotation=45)
plt.show()

# Log-transform skewed features to reduce impact of outliers
df['tobacco_log'] = np.log1p(df['tobacco'])
df['alcohol_log'] = np.log1p(df['alcohol'])

# Remove original versions of log-transformed features
df.drop(columns=['tobacco', 'alcohol'], inplace=True)

# Standardize continuous features to improve model performance
features_to_scale = ['sbp', 'tobacco_log', 'ldl', 'adiposity', 'age', 'alcohol_log']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Drop feature with lower correlation to the target
df.drop(columns=['typea'], inplace=True)

# Separate features and target variable
X = df.drop(columns=['chd'])
y = df['chd']

# Split data using stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define function to train and evaluate models
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print performance results
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
                xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD'])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model  # Return the trained model

# Logistic Regression with Ridge (L2) penalty and hyperparameter tuning
log_reg = GridSearchCV(
    LogisticRegression(penalty='l2', solver='liblinear'),
    param_grid={'C': [0.1, 1, 10]}, cv=5
)
log_reg = train_evaluate_model(log_reg, X_train, y_train, X_test, y_test, "Ridge Logistic Regression")

# Random Forest (baseline model)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf = train_evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")

# Support Vector Machine with class weighting and hyperparameter tuning
svm_grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced', random_state=42),
    param_grid={'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']},
    cv=5, scoring='f1', n_jobs=-1
)
svm = train_evaluate_model(svm_grid, X_train, y_train, X_test, y_test, "Support Vector Machine")

# K-Nearest Neighbors (distance-based classifier)
knn = KNeighborsClassifier(n_neighbors=5)
knn = train_evaluate_model(knn, X_train, y_train, X_test, y_test, "K-Nearest Neighbors")

# Compare classifiers using ROC curve and AUC
plt.figure(figsize=(8, 6))

for model, name in zip([log_reg.best_estimator_, rf, svm, knn], ["Logistic Regression", "Random Forest", "SVM", "KNN"]):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

# Plot baseline
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Identify and print the best performing model by accuracy
best_model = max(
    [log_reg.best_estimator_, rf, svm, knn],
    key=lambda m: accuracy_score(y_test, m.predict(X_test))
)
print(f"\nThe best performing model is: {best_model.__class__.__name__}")
