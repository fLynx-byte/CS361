import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
column_names = ["age", "year", "nodes", "survival_status"]
data = pd.read_csv("/Users/raymond/CS361/Homework10/Question3/haberman.data", header=None, names=column_names)

# Split the dataset into features (X) and labels (y)
X = data.drop("survival_status", axis=1)
y = data["survival_status"]

# Split the data into training and evaluation sets (80-20)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a decision tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=50, random_state=42)
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_eval)

# Build a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_eval)

# Evaluate the classifiers
dt_confusion_matrix = confusion_matrix(y_eval, dt_y_pred)
rf_confusion_matrix = confusion_matrix(y_eval, rf_y_pred)

dt_accuracy = accuracy_score(y_eval, dt_y_pred)
rf_accuracy = accuracy_score(y_eval, rf_y_pred)

print("Decision Tree Classifier Confusion Matrix:")
print(dt_confusion_matrix)
print(f"Decision Tree Classifier Accuracy: {dt_accuracy}\n")

print("Random Forest Classifier Confusion Matrix:")
print(rf_confusion_matrix)
print(f"Random Forest Classifier Accuracy: {rf_accuracy}")


