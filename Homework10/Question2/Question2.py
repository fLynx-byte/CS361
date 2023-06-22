import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
from scipy import stats
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.neighbors._classification", lineno=228)

data = pd.read_csv("/Users/raymond/CS361/Homework10/Question2/EEG Eye State.arff", comment='@', header=None)

data.columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'eyeDetection']

# Split the data into train and test sets
X = data.drop('eyeDetection', axis=1)
y = data['eyeDetection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=50, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Decision Tree evaluation
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_confusion = confusion_matrix(y_test, dt_pred)

# Random Forest evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_confusion = confusion_matrix(y_test, rf_pred)

print("Decision Tree Accuracy: ", dt_accuracy)
print("Decision Tree Confusion Matrix: \n", dt_confusion)
print("Random Forest Accuracy: ", rf_accuracy)
print("Random Forest Confusion Matrix: \n", rf_confusion)


# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Support Vector Machine
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# k-NN evaluation
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_confusion = confusion_matrix(y_test, knn_pred)

# SVM evaluation
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_confusion = confusion_matrix(y_test, svm_pred)

print("k-NN Accuracy: ", knn_accuracy)
print("k-NN Confusion Matrix: \n", knn_confusion)
print("SVM Accuracy: ", svm_accuracy)
print("SVM Confusion Matrix: \n", svm_confusion)

# Cross-validation
models = [dt, rf, knn, svm]
model_names = ['Decision Tree', 'Random Forest', 'k-NN', 'SVM']
cv_scores = []

for model in models:
    scores = cross_val_score(model, X, y, cv=5)
    cv_scores.append(scores.mean())

best_model_index = cv_scores.index(max(cv_scores))
best_model = models[best_model_index]
best_model_name = model_names[best_model_index]

print("Best model: ", best_model_name)
print("Best model accuracy: ", cv_scores[best_model_index])

# Fitting the best model on the whole dataset and getting the confusion matrix
best_model.fit(X_train, y_train)
best_model_pred = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_test, best_model_pred)
best_model_confusion = confusion_matrix(y_test, best_model_pred)

print("Best model accuracy: ", best_model_accuracy)
print("Best model confusion matrix: \n", best_model_confusion)


print("k-NN Accuracy: ", knn_accuracy)
print("k-NN Confusion Matrix: \n", knn_confusion)
print("SVM Accuracy: ", svm_accuracy)
print("SVM Confusion Matrix: \n", svm_confusion)
print("Best model: ", best_model_name)
print("Best model accuracy: ", cv_scores[best_model_index])
print("Best model confusion matrix: \n", best_model_confusion)


