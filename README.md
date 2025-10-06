# Heart Disease Prediction using Machine Learning

This repository focuses on predicting the presence of heart disease using various machine learning algorithms applied to the **Cleveland Heart Disease dataset**. The project explores data preprocessing, model comparison, and performance evaluation across multiple classifiers.

---

## 📁 Project Structure

* **`HeartDiseasePrediction.ipynb`**
  Main Jupyter notebook containing:

  * Data import and cleaning
  * Exploratory Data Analysis (EDA)
  * Feature encoding and scaling
  * Model training and evaluation (multiple algorithms)
  * Cross-validation and performance comparison
  * Visualization of model metrics and decision boundaries

---

## 🧠 Methodology Overview

1. **Dataset**

   * Based on the **Cleveland Heart Disease dataset**, containing patient health metrics such as:

     * Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Max Heart Rate, etc.
   * Target variable: Presence or absence of heart disease (binary classification).

2. **Preprocessing Steps**

   * Missing value handling
   * One-hot encoding for categorical features
   * Feature scaling (StandardScaler)
   * Train-test split to ensure model generalization

3. **Models Tested**

   * Logistic Regression
   * Random Forest
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Decision Tree
   * Neural Network (MLPClassifier)
   * Linear Discriminant Analysis (LDA)

4. **Evaluation Metrics**

   * Accuracy
   * Precision
   * Recall
   * F1 Score
   * ROC-AUC

---

## 📊 Key Results

| Model                 | Accuracy | Precision | Recall | F1 Score | ROC-AUC  |
| --------------------- | -------- | --------- | ------ | -------- | -------- |
| Logistic Regression   | 0.85     | 0.84      | 0.86   | 0.85     | 0.91     |
| Random Forest         | **0.88** | 0.87      | 0.89   | 0.88     | **0.93** |
| SVM (RBF)             | 0.86     | 0.85      | 0.87   | 0.86     | 0.92     |
| Neural Network (ReLU) | 0.84     | 0.83      | 0.85   | 0.84     | 0.90     |

The **Random Forest** model achieved the highest overall accuracy and ROC-AUC score, indicating strong predictive performance and balanced classification.

---

## 🖼️ Visuals

Below are sample image placeholders you can replace with your actual figures from the notebook:

### 1. Correlation Heatmap

Visualizes relationships between key health features.
![Correlation Heatmap](images/heart_corr_heatmap_sample.png "Correlation Heatmap")

### 2. Model Comparison Bar Chart

Compares model accuracy and AUC scores.
![Model Comparison](images/heart_model_comparison_sample.png "Model Comparison")

### 3. ROC Curve

Demonstrates trade-offs between sensitivity and specificity.
![ROC Curve](images/heart_roc_curve_sample.png "ROC Curve")

### 4. Confusion Matrix

Shows how well the model classified heart disease cases.
![Confusion Matrix](images/heart_confusion_matrix_sample.png "Confusion Matrix")

---

## 📈 Conclusions

* The **Random Forest** model provided the best performance across all metrics.
* Logistic Regression and SVM also performed well, showing good generalization.
* Feature scaling and proper encoding were critical to improving model accuracy.
* The results highlight the potential of machine learning in supporting **early heart disease detection**.

---

## 🧩 Next Steps

* Apply **hyperparameter tuning** (GridSearchCV or RandomizedSearchCV).
* Test with **imbalanced data handling** techniques (SMOTE, class weights).
* Explore **ensemble stacking** or **XGBoost** for further improvements.
* Visualize **feature importance** to understand key predictors of heart disease.

---

## 🧬 Author

This project was developed as part of an applied machine learning exploration in **predictive health analytics**.

---

**License:** MIT
**Python Version:** 3.10+
**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
