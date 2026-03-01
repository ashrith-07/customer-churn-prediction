"""
Customer Churn Prediction - Model Training Module
Trains and evaluates Logistic Regression and Decision Tree models.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Starting model training and evaluation...")

# Load preprocessed arrays and transformation pipeline
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')
y_test  = np.load('data/y_test.npy')
preprocessor = joblib.load('models/preprocessor.pkl')

feature_names = preprocessor.get_feature_names_out()

# Initialize and train Logistic Regression
print("\nEvaluating Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, lr_preds):.4f}")
print(f"Precision: {precision_score(y_test, lr_preds):.4f}")
print(f"Recall:    {recall_score(y_test, lr_preds):.4f}")
print(f"F1 Score:  {f1_score(y_test, lr_preds):.4f}")

# Initialize and train Decision Tree
print("\nEvaluating Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, dt_preds):.4f}")
print(f"Precision: {precision_score(y_test, dt_preds):.4f}")
print(f"Recall:    {recall_score(y_test, dt_preds):.4f}")
print(f"F1 Score:  {f1_score(y_test, dt_preds):.4f}")

# Extract top churn drivers from Decision Tree feature importances
print("\nTop 3 Churn Drivers:")
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(3):
    print(f"- {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Generate and save Visualizations for the LaTeX report
# 1. Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual Churn')
plt.xlabel('Predicted Churn')
plt.savefig('confusion_matrix.png')
plt.close() # Closes the figure so it doesn't overlap with the next one

# 2. Decision Tree Plot
plt.figure(figsize=(25,10)) # Made very large so the text is readable
plot_tree(dt_model, filled=True, feature_names=feature_names, class_names=['Stayed', 'Churned'], rounded=True, fontsize=10)
plt.title('Decision Tree - Customer Churn Drivers')
plt.savefig('decision_tree.png')
plt.close()

print("\nVisualizations saved as 'confusion_matrix.png' and 'decision_tree.png' (Ready for LaTeX).")

# Serialize trained models for UI deployment
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(dt_model, 'models/dt_model.pkl')

print("Model training complete. Serialized models saved to 'models/' directory.")