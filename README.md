# Online Shoppers Purchasing Intention

This project focuses on predicting whether an online shopper will make a purchase based on their session behavior. Using a dataset of user sessions, the project implements machine learning techniques to address an imbalanced classification problem.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
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

---

## Features

### Numerical Features
- **Administrative** / **Administrative Duration**: Number of administrative pages visited and total time spent on them.
- **Informational** / **Informational Duration**: Number of informational pages visited and total time spent on them.
- **ProductRelated** / **ProductRelated Duration**: Number of product-related pages visited and total time spent on them.
- **BounceRate**: Percentage of visitors who leave after viewing a single page.
- **ExitRate**: Percentage of pageviews that result in an exit from the site.
- **PageValues**: Average value of a web page a user visited before completing an e-commerce transaction.
- **SpecialDay**: Closeness of the session to a special day (e.g., Valentine's Day).

### Categorical Features
- **Month**: Month of the session.
- **OperatingSystems**: Operating system of the user.
- **Browser**: Browser used during the session.
- **Region**: User's geographic region.
- **TrafficType**: Traffic source type.
- **VisitorType**: Whether the visitor is new or returning.
- **Weekend**: Boolean indicating if the session occurred on a weekend.

### Target Feature
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
   - Addressing class imbalance with **SMOTEENN** (SMOTE for oversampling + ENN for under-sampling).

4. **Feature Selection**:
   - Using the chi-squared test to identify the most impactful features.

5. **Model Building and Tuning**:
   - Training models with K-Fold cross-validation.
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
- **Random Forest**:
  - Best score: 0.6717 (original data), 0.8328 (resampled data).
  - Number of features: 12 (original), 4 (resampled).

- **Decision Tree**:
  - Best score: 0.6714 (original data), 0.9266 (resampled data).
  - Number of features: 2 (original), 8 (resampled).

- **XGBoost**:
  - Best score: 0.6876 (original data), 0.9264 (resampled data).
  - Number of features: 48 (original), 12 (resampled).

- **KNN**:
  - Best score: 0.5151 (original data), 0.9779 (resampled data).
  - Number of features: 2 (original), 65 (resampled).

Class balancing through SMOTEENN significantly improved model performance, especially for recall and precision.

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
   cd online-shoppers-intention
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