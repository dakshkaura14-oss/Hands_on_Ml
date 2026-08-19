# Wine Dataset — Exploratory Data Analysis & Classification

## 📌 Overview

This project explores the **Wine dataset** and builds machine learning classification models using Scikit-learn.

The main objective is to understand the dataset through **Exploratory Data Analysis (EDA)**, perform feature engineering, build preprocessing pipelines, and compare different classification algorithms using cross-validation.

## 📊 Dataset

The project uses the Wine dataset available through Scikit-learn.

The dataset contains **178 wine samples** belonging to **3 different wine classes** with **13 numerical features** describing the chemical properties of the wines.

Some of the features include:

* Alcohol
* Malic acid
* Ash
* Alcalinity of ash
* Magnesium
* Total phenols
* Flavanoids
* Nonflavanoid phenols
* Proanthocyanins
* Color intensity
* Hue
* OD280/OD315 of diluted wines
* Proline

The `target` column represents the wine class.

## 🔍 Exploratory Data Analysis

EDA was performed to understand relationships between the features and identify potentially useful features.

The analysis included:

* Feature distributions
* Correlation analysis
* Scatter plots
* Scatter plot matrices
* Feature relationship analysis
* Identification of potentially useful feature combinations

One feature-engineering experiment involved creating the ratio:

```text
Alcalinity of ash / Flavanoids
```

This ratio was added using a custom Scikit-learn transformer.

## ⚙️ Data Splitting

A **StratifiedShuffleSplit** was used to divide the dataset into training and testing sets while maintaining approximately the same class distribution in both sets.

```python
StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=24
)
```

## 🔧 Preprocessing Pipeline

A Scikit-learn `Pipeline` was used to combine preprocessing and model training.

The preprocessing workflow includes:

```text
Input Data
    ↓
Median Imputation
    ↓
Feature Engineering (optional)
    ↓
Standard Scaling
    ↓
Classification Model
```

Using a pipeline ensures that preprocessing is consistently applied during training, validation, and testing.

## 🤖 Models

### Logistic Regression

Logistic Regression was evaluated with and without the engineered ratio feature.

The cross-validation results were approximately:

| Configuration                       | 5-Fold CV Accuracy |
| ----------------------------------- | -----------------: |
| Logistic Regression                 |             98.32% |
| Logistic Regression + Ratio Feature |         **98.89%** |

The ratio feature therefore produced a **small improvement** in cross-validation accuracy.

### K-Nearest Neighbors

KNN was also evaluated with feature scaling and different values of `k`.

Several values were tested:

```text
k = 1, 3, 5, 7, 9, 11, 13
```

The best observed cross-validation accuracy was approximately:

```text
97.78%
```

for:

```text
k = 7, 9, 13
```

## 📈 Model Comparison

Based on the experiments:

| Model               | Best CV Accuracy |
| ------------------- | ---------------: |
| Logistic Regression |       **98.89%** |
| KNN                 |           97.78% |

Logistic Regression performed slightly better on this dataset.

The difference is relatively small, demonstrating that both models can perform well on the Wine dataset.

## 📉 Evaluation Metrics

The models were evaluated using:

### Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Log Loss

Log Loss evaluates the quality of the model's predicted probabilities.

A lower value indicates better probabilistic predictions.

The Logistic Regression model achieved a test log loss of approximately:

```text
0.0345
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / VS Code

## 🚀 Future Improvements

Potential improvements include:

* Hyperparameter tuning using `GridSearchCV`
* Testing additional classifiers such as SVM, Random Forest, and Gradient Boosting
* Comparing additional evaluation metrics
* Confusion matrix analysis
* Feature importance analysis
* Further feature engineering
* Model visualization and decision-boundary analysis

## 🎯 Key Takeaway

This project demonstrates a complete introductory machine-learning workflow:

```text
EDA
 ↓
Feature Analysis
 ↓
Train/Test Split
 ↓
Stratified Sampling
 ↓
Custom Feature Engineering
 ↓
Preprocessing Pipeline
 ↓
Model Training
 ↓
Cross-Validation
 ↓
Hyperparameter Tuning
 ↓
Model Comparison
```

The experiments show that **Logistic Regression performs particularly well on the Wine dataset**, achieving approximately **98.89% cross-validation accuracy** with the engineered feature.
