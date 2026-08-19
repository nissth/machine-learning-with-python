# Machine Learning with Python

A comprehensive collection of machine learning algorithms and techniques implemented in Python, covering everything from data preprocessing to deep learning. This repository was built as part of the **"Python ile Makine Öğrenmesi"** course by **Şadi Evren Şeker**.

---

## Repository Structure

```
machine-learning-with-python/
├── data-preprocessing/
├── prediction/
├── classification/
├── clustering/
├── dimension-reduction/
├── association-rule-mining/
├── model-evaluation/
├── natural-language-processing/
├── reinforced-learning/
├── deep-learning/
└── image-processing/
```

---

##  Topics Covered

###  Data Preprocessing
Fundamental data preparation techniques before feeding data into any ML model.

| File | Description |
|------|-------------|
| `data-import.py` | Loading datasets from CSV files |
| `data-frame.py` | Working with Pandas DataFrames |
| `missing-datas.py` | Handling missing values with imputation |
| `categorical-datas.py` | Encoding categorical variables (Label & One-Hot Encoding) |
| `split-dataset.py` | Splitting data into training and test sets |
| `standardization.py` | Feature scaling and standardization |

---

###  Prediction (Regression)
Supervised learning methods for predicting continuous numerical outputs.

| File / Folder | Algorithm |
|---------------|-----------|
| `linear-regression.py` | Simple Linear Regression |
| `multiple-linear-regression.py` | Multiple Linear Regression |
| `backward-elimination.py` | Feature selection via Backward Elimination |
| `polynomial-regression/` | Polynomial Regression |
| `support-vector-regression/` | Support Vector Regression (SVR) |
| `decision-tree/` | Decision Tree Regression |
| `random-forest/` | Random Forest Regression |
| `salary-example/` | End-to-end salary prediction example |
| `tennis-example/` | Decision-making example with tennis data |
| `pickle-model-save/` | Saving and loading models with Pickle |

---

###  Classification
Supervised learning methods for predicting categorical class labels.

| File / Folder | Algorithm |
|---------------|-----------|
| `logistic-regression/` | Logistic Regression |
| `k-nearest-neighbors/` | K-Nearest Neighbors (KNN) |
| `support-vector-machine/` | Support Vector Machine (SVM) |
| `naive-bayes/` | Naive Bayes Classifier |
| `decision-tree/` | Decision Tree Classification |
| `random-forest/` | Random Forest Classification |
| `iris-example/` | Classic Iris dataset classification example |

---

###  Clustering
Unsupervised learning methods for grouping similar data points.

| Folder | Algorithm |
|--------|-----------|
| `k-means/` | K-Means Clustering |
| `hierarchy/` | Hierarchical (Agglomerative) Clustering |

---

###  Dimension Reduction
Techniques to reduce the number of features while preserving information.

| Folder | Algorithm |
|--------|-----------|
| `principal-component-analysis/` | PCA — Principal Component Analysis |
| `linear-discriminant-analysis/` | LDA — Linear Discriminant Analysis |

---

###  Association Rule Mining
Market basket analysis and frequent itemset discovery.

| Folder | Algorithm |
|--------|-----------|
| `apriori/` | Apriori Algorithm |
| `eclat/` | ECLAT Algorithm |

---

###  Model Evaluation
Methods to assess and optimize model performance.

| Folder | Technique |
|--------|-----------|
| `k-fold-cross-validation/` | K-Fold Cross Validation |
| `grid-search/` | Hyperparameter Tuning with Grid Search |

---

###  Natural Language Processing
Text processing and sentiment analysis techniques.

| File | Description |
|------|-------------|
| `nlp.py` | Bag of Words model for restaurant review sentiment analysis |

**Dataset:** `Restaurant_Reviews.csv`

---

###  Reinforced Learning
Exploration-exploitation strategies for sequential decision making.

| Folder | Algorithm |
|--------|-----------|
| `upper-confidence-bound/` | UCB — Upper Confidence Bound |
| `thompson-sampling/` | Thompson Sampling |

**Dataset:** `Ads_CTR_Optimisation.csv`

---

###  Deep Learning
Neural network architectures for complex pattern recognition.

| File | Description |
|------|-------------|
| `ann.py` | Artificial Neural Network (ANN) for customer churn prediction |
| `xgboost_code.py` | XGBoost classifier |
| `test.py` | Model testing and evaluation |

**Dataset:** `Churn_Modelling.csv`

---

###  Image Processing
Deep learning for image classification tasks.

| Folder | Architecture |
|--------|-------------|
| `convolutional-neural-network/` | CNN — Convolutional Neural Network |

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=matplotlib&logoColor=white)

**Core Libraries:**
- `numpy` — Numerical computing
- `pandas` — Data manipulation
- `matplotlib` / `seaborn` — Data visualization
- `scikit-learn` — Classical ML algorithms
- `tensorflow` / `keras` — Deep learning
- `xgboost` — Gradient boosting

---

##  Getting Started

### Prerequisites

Make sure you have Python 3.x installed. Then install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras xgboost
```

### Clone the Repository

```bash
git clone https://github.com/<nissth>/machine-learning-with-python.git
cd machine-learning-with-python
```

### Run an Example

```bash
# Example: Run K-Means Clustering
python clustering/k-means/k-means.py

# Example: Run Logistic Regression Classification
python classification/logistic-regression/logistic-regression.py

# Example: Run ANN for Churn Prediction
python deep-learning/ann.py
```

---

##  Course Information

This repository contains my notes and implementations from the course:

> **Python ile Makine Öğrenmesi**
> Instructor: **Şadi Evren Şeker**
> [BTK Akademi](https://www.btkakademi.gov.tr)

---

##  License

This project is open source and available under the [MIT License](LICENSE).

---

##  Author

Feel free to explore, fork, and star the repository if you found it helpful!
