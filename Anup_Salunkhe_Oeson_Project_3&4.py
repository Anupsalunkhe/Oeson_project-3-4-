#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install xgboost


# In[11]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


data = pd.read_csv(r"C:\Users\ANUP\Downloads\Student_academic.csv")


# In[18]:


# EDA (Project 3)
# 1. Data Overview
print(data.info())


# In[21]:


# Check for missing values
missing_values = data.isnull().sum()


# In[22]:


# 3. Descriptive Statistics
print(data.describe())


# In[23]:


# Check for duplicate rows
duplicate_rows = data[data.duplicated()]
print("\nDuplicate Rows:")
print(duplicate_rows)


# In[24]:


data.duplicated()


# In[25]:


data.isnull().sum()


# # 4. Data Visualization

# In[26]:


# 4. Data Visualization
# Plot charts to visualize data distributions
# Example: Histogram of Age at enrollment
sns.histplot(data['Age at enrollment'])
plt.title('Distribution of Age at Enrollment')
plt.show()


# In[27]:


# Example: Countplot of Marital Status by Target Variable
sns.countplot(x='Marital status', hue='Target', data=data)
plt.title('Countplot of Marital Status by Target Variable')
plt.show()


# In[28]:


# Example: Countplot of Application Mode by Target Variable
sns.countplot(x='Application mode', hue='Target', data=data)
plt.title('Countplot of Application Mode by Target Variable')
plt.show()


# In[29]:


# Example: Pie Chart of Target Variable
target_distribution = data['Target'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(target_distribution, labels=target_distribution.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen', 'lightskyblue'])
plt.title('Distribution of Target Variable')
plt.show()


# In[30]:


# Define custom colors for each target category
colors = {'dropout': 'blue', 'graduate': 'green', 'enrolled': 'red'}

# Create histograms
plt.figure(figsize=(12, 8))

# Plot histograms for each target category with custom colors
for target_category in data['Target'].unique():
    subset_data = data[data['Target'] == target_category]
    
    # Convert target category to lowercase to match the keys in the 'colors' dictionary
    target_category_lower = target_category.lower()
    
    plt.hist(subset_data['Unemployment rate'], bins=20, alpha=0.5, label=target_category, color=colors[target_category_lower])

plt.title('Distribution of Unemployment Rate by Target Variable')
plt.xlabel('Unemployment Rate')
plt.ylabel('Count')
plt.legend()
plt.show()


# In[31]:


# Handle NaN values in the 'International' column (if any)
data['International'].fillna(0, inplace=True)

# Create a count plot
plt.figure(figsize=(10, 6))
sns.countplot(x='Target', hue='International', data=data, palette=['lightblue', 'lightcoral'])
plt.title('Count Plot of International and Non-International Students by Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.legend(title='International', labels=['Non-International', 'International'])
plt.show()


# In[32]:


# Handle NaN values in the 'Gender' column (if any)
data['Gender'].fillna(0, inplace=True)

# Create a violin plot with legend
plt.figure(figsize=(12, 8))
sns.violinplot(x='Target', y='Gender', data=data, palette=['lightblue', 'lightcoral', 'lightskyblue'], hue_order=[0, 1])
plt.title('Violin Plot of Gender Distribution by Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Gender (0-Female, 1-Male)')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()


# In[33]:


# Example: Histogram for Age at Enrollment by Target Variable
# Assuming 'Age at enrollment' is a numerical column in your dataset

# Handle NaN values in the 'Age at enrollment' column (if any)
data['Age at enrollment'].fillna(data['Age at enrollment'].mean(), inplace=True)

# Create histograms
plt.figure(figsize=(12, 8))

# Plot histograms for each target category
for target_category in data['Target'].unique():
    subset_data = data[data['Target'] == target_category]
    plt.hist(subset_data['Age at enrollment'], bins=20, alpha=0.5, label=target_category)

plt.title('Distribution of Age at Enrollment by Target Variable')
plt.xlabel('Age at Enrollment')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[34]:


# Mapping numerical codes to meaningful labels
marital_status_mapping = {
    1: 'Single',
    2: 'Married',
    3: 'Divorced',
    4: 'Widowed',
    5: 'Separated',
    6: 'Other/Unknown'
}

# Replace numerical codes with meaningful labels
data['Marital status'] = data['Marital status'].map(marital_status_mapping)

# Handle NaN values in the 'Marital status' column (if any)
data['Marital status'].fillna('Other/Unknown', inplace=True)

# Create a bar plot
plt.figure(figsize=(12, 8))
sns.countplot(x='Marital status', hue='Target', data=data, palette='muted')
plt.title('Bar Plot of Marital Status by Target Variable')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(title='Target', labels=data['Target'].unique())
plt.show()


# In[66]:


# Select numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Determine the number of rows and columns for subplots
num_cols = len(numerical_columns)
num_rows = (num_cols - 1) // 3 + 1

# Plot histograms for numerical features
plt.figure(figsize=(15, 5 * num_rows))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(num_rows, 3, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# # ML PART

# # Decision Tree

# In[51]:


# Assuming 'Target' is the column you want to predict
X = data.drop('Target', axis=1)
y = data['Target']

# Encode the target variable to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),  # Increase max_iter
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate models
results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
 # Assuming y_pred_prob is the probability estimates for all classes
y_pred_prob = pipeline.predict_proba(X_test)

# Extract the probability estimates for the positive class (or a specific class)
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')


results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC AUC': roc_auc,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    })

# Display results
for result in results:
    print(f"\nResults for {result['Model']}:\n")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"ROC AUC: {result['ROC AUC']:.4f}")
    print("\nConfusion Matrix:")
    print(result['Confusion Matrix'])
    print("\nClassification Report:")
    print(result['Classification Report'])

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {result["Model"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# # Logistic Regression 

# In[52]:


# Assuming 'Target' is the column you want to predict
X = data.drop('Target', axis=1)
y = data['Target']

# Encode the target variable to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed

# Create a pipeline with preprocessing and Logistic Regression model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', logistic_regression_model)])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Assuming y_pred_prob is the probability estimates for all classes
y_pred_prob = pipeline.predict_proba(X_test)

# Extract the probability estimates for the positive class (or a specific class)
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

# Display results
print(f"\nResults for Logistic Regression:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # K-Nearest Neighbors (KNN) 

# In[53]:


# Assuming 'Target' is the column you want to predict
X = data.drop('Target', axis=1)
y = data['Target']

# Encode the target variable to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Define K-Nearest Neighbors model
knn_model = KNeighborsClassifier()

# Create a pipeline with preprocessing and K-Nearest Neighbors model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', knn_model)])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Assuming y_pred_prob is the probability estimates for all classes
y_pred_prob = pipeline.predict_proba(X_test)

# Extract the probability estimates for the positive class (or a specific class)
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

# Display results
print(f"\nResults for K-Nearest Neighbors:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for K-Nearest Neighbors')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Random Forest

# In[54]:


# Assuming 'Target' is the column you want to predict
X = data.drop('Target', axis=1)
y = data['Target']

# Encode the target variable to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Define Random Forest model
random_forest_model = RandomForestClassifier()

# Create a pipeline with preprocessing and Random Forest model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', random_forest_model)])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Assuming y_pred_prob is the probability estimates for all classes
y_pred_prob = pipeline.predict_proba(X_test)

# Extract the probability estimates for the positive class (or a specific class)
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

# Display results
print(f"\nResults for Random Forest:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Gradient Booster 

# In[55]:


# Assuming 'Target' is the column you want to predict
X = data.drop('Target', axis=1)
y = data['Target']

# Encode the target variable to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Train the Gradient Boosting model
model = GradientBoostingClassifier()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Assuming y_pred_prob is the probability estimates for all classes
y_pred_prob = pipeline.predict_proba(X_test)

# Extract the probability estimates for the positive class (or a specific class)
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

# Display results
print("\nResults for Gradient Boosting:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[70]:


from prettytable import PrettyTable

# Assuming 'Target' is the column you want to predict
X = data.drop('Target', axis=1)
y = data['Target']

# Encode the target variable to numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate models
results = PrettyTable()
results.field_names = ["Model", "Accuracy", "ROC AUC"]

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Assuming y_pred_prob is the probability estimates for all classes
    y_pred_prob = pipeline.predict_proba(X_test)
    
    # Extract the probability estimates for the positive class (or a specific class)
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    
    results.add_row([name, f"{accuracy_score(y_test, y_pred):.4f}", f"{roc_auc:.4f}"])

print(results)


# In[71]:


import matplotlib.pyplot as plt
import numpy as np

# Data
model_names = ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Gradient Boosting"]
accuracies = [0.7525, 0.7051, 0.6802, 0.7695, 0.7571]

# Define colors for each bar
colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral', 'orchid']

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=colors)

# Adding data values on top of each bar
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{accuracy:.4f}', fontsize=8)

plt.title('Accuracy Comparison of Machine Learning Models')
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Setting y-axis limits to represent accuracy values between 0 and 1
plt.show()


# In[73]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Train and evaluate models
results = []
roc_curves = {}  # Dictionary to store ROC curves

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred_prob = pipeline.predict_proba(X_test)
    
    # Compute ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=[0, 1, 2])[:, 0], y_pred_prob[:, 0])
    roc_auc = auc(fpr, tpr)
    
    # Store ROC curve values
    roc_curves[name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC AUC': roc_auc,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    })

# Display results
for result in results:
    print(f"\nResults for {result['Model']}:\n")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"ROC AUC: {result['ROC AUC']:.4f}")
    print("\nConfusion Matrix:")
    print(result['Confusion Matrix'])
    print("\nClassification Report:")
    print(result['Classification Report'])

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {result["Model"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))

for name, curve in roc_curves.items():
    plt.plot(curve['fpr'], curve['tpr'], label=f'{name} (AUC = {curve["roc_auc"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Models')
plt.legend(loc="lower right")
plt.show()


# # Cross Validation For All The ML Models

# In[74]:


from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define the number of folds for cross-validation
n_folds = 5

# Define a cross-validation strategy (StratifiedKFold for classification problems)
cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Results list to store the cross-validation results for each model
results = []

# Iterate over models
for name, model in models.items():
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Perform cross-validation
    cv_results = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring='accuracy')
    
    # Append the results to the list
    results.append({
        'Model': name,
        'Cross-Validation Accuracy': cv_results,
        'Average Accuracy': cv_results.mean(),
        'Accuracy Standard Deviation': cv_results.std()
    })

# Display results
for result in results:
    print(f"\nResults for {result['Model']}:\n")
    print(f"Cross-Validation Accuracy: {result['Average Accuracy']:.4f} ± {result['Accuracy Standard Deviation']:.4f}")
    print(f"Individual Fold Accuracies: {result['Cross-Validation Accuracy']}")


# In summary, both Logistic Regression and Gradient Boosting show good and stable performance. Random Forest also performs well, while K-Nearest Neighbors has slightly higher variability. The Decision Tree model appears to have lower and more variable accuracy, indicating room for improvement or consideration of alternative models.

# # Conclusion

# Model Performance:
# 
# Gradient Boosting and Random Forest: These models consistently show strong performance across multiple metrics, including accuracy and ROC AUC. They seem to be well-suited for the given task.
# 
# Logistic Regression: While having a decent accuracy and ROC AUC, it's slightly below the performance of Gradient Boosting and Random Forest.
# 
# K-Nearest Neighbors: It has a lower accuracy compared to Gradient Boosting and Random Forest. Consider exploring further optimization or other models.
# 
# Decision Tree: Shows the lowest accuracy among the models. There might be room for improvement or considering more complex models.
# 
# Metric Consideration:
# 
# Accuracy: It provides a general overview of correct predictions but might not be the sole metric to rely on, especially if there is class imbalance.
# 
# ROC AUC: Useful for evaluating the model's ability to distinguish between classes. Higher ROC AUC values indicate better discrimination.
# 
# Further Steps:
# 
# Model Fine-Tuning: Consider fine-tuning hyperparameters for Gradient Boosting, Random Forest, and other models to potentially improve performance.
# 
# Feature Engineering: Evaluate if additional feature engineering could enhance model performance.
# 
# Imbalanced Classes: If the dataset has imbalanced classes, consider techniques like oversampling, undersampling, or using different evaluation metrics.
# 
# Overall Recommendation:
# 
# Gradient Boosting and Random Forest: These models seem promising and are recommended for further exploration and potential deployment.

# # Cross validation

# Logistic Regression:
# 
# Cross-Validation Accuracy: 0.7631 ± 0.0112
# Individual Fold Accuracies: [0.774, 0.773, 0.768, 0.755, 0.745]
# Explanation: The model shows a consistent accuracy around 76-77% in different folds. The small standard deviation (±0.0112) indicates relatively low variability, suggesting the model is stable and performs consistently.
# K-Nearest Neighbors:
# 
# Cross-Validation Accuracy: 0.6987 ± 0.0122
# Individual Fold Accuracies: [0.706, 0.689, 0.697, 0.718, 0.683]
# Explanation: The model has an accuracy around 70%, with a slightly higher standard deviation (±0.0122). It may exhibit more variability in performance across different folds.
# Gradient Boosting:
# 
# Cross-Validation Accuracy: 0.7724 ± 0.0044
# Individual Fold Accuracies: [0.777, 0.776, 0.774, 0.767, 0.767]
# Explanation: The model shows a high and consistent accuracy around 77%, with a very low standard deviation (±0.0044). This suggests a stable and robust performance.
# Random Forest:
# 
# Cross-Validation Accuracy: 0.7694 ± 0.0071
# Individual Fold Accuracies: [0.760, 0.773, 0.774, 0.762, 0.778]
# Explanation: The model has a good accuracy around 77%, with a moderate standard deviation (±0.0071). It appears stable with acceptable variability.
# Decision Tree:
# 
# Cross-Validation Accuracy: 0.6761 ± 0.0203
# Individual Fold Accuracies: [0.638, 0.680, 0.699, 0.685, 0.678]
# Explanation: The model has a lower accuracy around 68%, and the higher standard deviation (±0.0203) suggests more variability. It may benefit from more tuning or could be inherently less stable.

# # Over all Conclusion

# Based on both the normal evaluation and cross-validation, it is evident that Logistic Regression, Gradient Boosting, and Random Forest are the most promising models for the given task. These models consistently exhibit competitive and robust performance across various metrics. Logistic Regression demonstrates a good balance between simplicity and performance, while Gradient Boosting and Random Forest showcase superior accuracy and ROC AUC.
# 
# Further efforts can be directed towards fine-tuning hyperparameters, exploring additional feature engineering, and carefully addressing any potential class imbalances in the dataset. While these models demonstrate promising results, it's essential to consider the specific requirements and constraints of the application before making a final model selection.
# 
# In summary, Logistic Regression, Gradient Boosting, and Random Forest are recommended for further exploration and potential deployment, with a preference for Gradient Boosting and Random Forest for their superior overall performance.

# # Thank You

# In[ ]:




