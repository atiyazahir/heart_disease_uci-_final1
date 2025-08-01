# heart_disease_uci-_final1
# 🫀 Heart Disease Prediction using Machine Learning

This project applies machine learning techniques to predict the presence of heart disease using the [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease). Multiple supervised learning models are trained, evaluated, and compared for performance.

---

## 📁 Dataset Overview

- **Source:** UCI Machine Learning Repository
- **Features:** 13-15 clinical features such as age, sex, cholesterol, blood pressure, etc.
- **Target:** Presence (1) or absence (0) of heart disease

---

## 🛠️ Preprocessing Steps

- Handling missing values using `SimpleImputer`
- Encoding categorical variables using `LabelEncoder`
- Feature scaling using `StandardScaler`
- Splitting dataset into training and testing sets

---

## 🤖 Machine Learning Models Applied

The following 10 ML models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Gradient Boosting Classifier
8. AdaBoost Classifier
9. XGBoost Classifier
10. LightGBM Classifier

---

## 📊 Evaluation Metrics

- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**
- **Comparison Bar Graph of Model Accuracies**

---

## 🔍 Example Usage

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
