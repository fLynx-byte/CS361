from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Support Vector Machine (SVM)
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Cross-validation
cv_dt = cross_val_score(dt, X, y, cv=5)
cv_rf = cross_val_score(rf, X, y, cv=5)
cv_svm = cross_val_score(svm, X, y, cv=5)
cv_knn = cross_val_score(knn, X, y, cv=5)

# Calculate the average accuracy for each model
avg_acc_dt = cv_dt.mean()
avg_acc_rf = cv_rf.mean()
avg_acc_svm = cv_svm.mean()
avg_acc_knn = cv_knn.mean()

# Find the best model
best_model_index = np.argmax([avg_acc_dt, avg_acc_rf, avg_acc_svm, avg_acc_knn])
models = [dt, rf, svm, knn]
best_model = models[best_model_index]

# Calculate the confusion matrix and accuracy for the best model
best_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
best_confusion = confusion_matrix(y_test, best_pred)

print("Best Model Accuracy: ", best_accuracy)
print("Best Model Confusion Matrix: \n", best_confusion)
