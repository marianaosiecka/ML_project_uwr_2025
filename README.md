# Online Shoppers Purchasing Intention

This project focuses on predicting whether an online shopper will make a purchase based on their session behavior. Using a dataset of user sessions, the project implements machine learning techniques to address an imbalanced classification problem.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Implemented Models](#implemented-models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Contributors](#contributors)

## Project Overview
The objective is to predict the `Revenue` feature (binary: 1 = purchase, 0 = no purchase) based on 17 session-related features. These include numerical data on user behavior (e.g., `BounceRate`, `ExitRate`) and categorical data (e.g., `Month`, `VisitorType`).

### Problem Statement
Understanding whether a customer is likely to make a purchase is essential for e-commerce businesses to improve their marketing efforts and create better user experiences. However, dealing with an imbalanced dataset makes it tricky for traditional classification models to perform well.

---

## Dataset
The dataset contains:
- **12,330 user sessions**
- **84.5% negative class** (no purchase) and **15.5% positive class** (purchase)
- **Numerical features**: User behavior metrics (e.g., `ProductRelated`, `BounceRate`, `PageValues`)
- **Categorical features**: Session metadata (e.g., `Month`, `Region`, `VisitorType`)

### Data Source
The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

### Features

#### Numerical Features
- **Administrative** / **Administrative Duration**: Number of administrative pages visited and total time spent on them.
- **Informational** / **Informational Duration**: Number of informational pages visited and total time spent on them.
- **ProductRelated** / **ProductRelated Duration**: Number of product-related pages visited and total time spent on them.
- **BounceRate**: Percentage of visitors who leave after viewing a single page.
- **ExitRate**: Percentage of pageviews that result in an exit from the site.
- **PageValues**: Average value of a web page a user visited before completing an e-commerce transaction.
- **SpecialDay**: Closeness of the session to a special day (e.g., Valentine's Day).

#### Categorical Features
- **Month**: Month of the session.
- **OperatingSystems**: Operating system of the user.
- **Browser**: Browser used during the session.
- **Region**: User's geographic region.
- **TrafficType**: Traffic source type.
- **VisitorType**: Whether the visitor is new or returning.
- **Weekend**: Boolean indicating if the session occurred on a weekend.

#### Target Feature
- **Revenue**: Boolean indicating whether a purchase was made during the session.

---

## Approach
The project workflow includes:
1. **Data Analysis**:
   - Exploration of feature distributions and correlations.
   - Identification of imbalances and null values.

2. **Preprocessing**:
   - Encoding categorical features using one-hot encoding.
   - Removing outliers using the IQR method.
   - Scaling numerical features with MinMaxScaler.

3. **Resampling Techniques**:
   - Addressing class imbalance with **SMOTEENN** (SMOTE for oversampling + ENN for undersampling).

4. **Feature Selection**:
   - Using the SelectKBest with Mutual Information as the scoring function.

5. **Model Building and Tuning**:
   - Training models with 10-fold cross-validation.
   - Hyperparameter optimization using GridSearchCV.

---

## Implemented Models
The following classification models were used:
1. **Decision Tree**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**
4. **XGBoost**

Each model was trained on both the original and resampled datasets to compare performance.

---

## Evaluation
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Curve**
- **Precision-Recall Curve**

---

## Results
### Original data 
Number of K best features:
- Decision Tree: 2
- Random Forest: 54
- KNN: 5
- XGBoost: 48

Evaluation metrics:
| Model          | Accuracy | Recall   | Precision | F1 score |
|----------------|----------|----------|-----------|----------|
| Decision Tree  | 0.874049 | 0.803103 | 0.565136  | 0.663425 |
| Random Forest  | 0.874297 | 0.803103 | 0.565775  | 0.663866 |
| KNN            | 0.863381 | 0.531835 | 0.561265  | 0.546154 |
| XGBoost        | 0.889018 | 0.756019 | 0.614615  | 0.678023 |

![image](https://github.com/user-attachments/assets/e26debb6-6dde-43ba-a14e-be7d189151bb)


### Resampled data
Number of K best features:
- Decision Tree: 11
- Random Forest: 3
- KNN: 12
- XGBoost: 11

Evaluation metrics:
| Model          | Accuracy | Recall   | Precision | F1 score |
|----------------|----------|----------|-----------|----------|
| Random Forest  | 0.920418 | 0.889603 | 0.946617  | 0.917225 |
| Decision Tree  | 0.931362 | 0.913918 | 0.945770  | 0.929571 |
| XGBoost        | 0.932955 | 0.920347 | 0.943013  | 0.931542 |
| KNN            | 0.923397 | 0.937255 | 0.910782  | 0.923829 |

![image](https://github.com/user-attachments/assets/245958ff-55e6-47d2-932b-f5b67039c624)


### Conclusion
Class balancing through SMOTEENN significantly improved model performance.

**XGBoost outperformed** the other models with both the original and resampled data.

---

## Setup and Usage
### Prerequisites
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - matplotlib
  - seaborn

### Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/marianaosiecka/ML_project_uwr_2025.git
   cd ML_project_uwr_2025
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   - Open `project.ipynb` in Jupyter Notebook or VSCode.
   - Execute the cells sequentially to preprocess the data, train models, and evaluate results.

---

## Contributors
- **Mafalda Costa** (351255)
- **Mariana Carvalho** (351254)
- **Denys Tsebulia** (351322)
