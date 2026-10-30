# Customer Churn Prediction Using Artificial Neural Networks

## Overview
This project builds a binary classification model to predict whether a bank customer is likely to leave the bank (`Exited = 1`) or stay (`Exited = 0`). It uses customer attributes such as credit score, age, geography, gender, tenure, balance, product usage, account activity, and estimated salary.

## Objectives
- Explore and prepare customer data.
- Encode categorical variables and standardize numerical features.
- Build an Artificial Neural Network (ANN) using TensorFlow/Keras.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Project Structure
```text
Customer-Churn-Prediction/
├── data/
│   └── Churn_Modelling.csv
├── Customer_Churn_Prediction_GitHub.ipynb
├── requirements.txt
└── README.md
```

## Dataset
The notebook expects `Churn_Modelling.csv` at `data/Churn_Modelling.csv`. The dataset is not included in this repository; add it locally and follow the dataset's original license and usage terms.

The target column is `Exited`:
- `0` = customer stayed
- `1` = customer exited

## Workflow
1. Load and inspect the dataset.
2. Check duplicate records and target distribution.
3. Drop identifier columns (`RowNumber`, `CustomerId`, `Surname`).
4. Apply one-hot encoding to `Geography` and `Gender`.
5. Split the data into training and testing sets.
6. Standardize features using `StandardScaler`.
7. Train an ANN with two ReLU hidden layers and a sigmoid output layer.
8. Evaluate predictions using classification metrics and a confusion matrix.

## Run Locally
```bash
pip install -r requirements.txt
```
Place the CSV file in the `data/` directory, then open the notebook in Jupyter Notebook/JupyterLab and run all cells.

## Conclusion
This project demonstrates an end-to-end customer churn prediction workflow using an Artificial Neural Network. It covers data preparation, feature engineering, scaling, model training, and evaluation. Since churn is the minority class, recall and F1-score for customers who exited are important alongside overall accuracy. The model can be improved further through threshold tuning, class-imbalance handling, cross-validation, hyperparameter tuning, and comparison with baseline models such as Logistic Regression and Random Forest.

## Future Improvements
- Compare ANN performance with Logistic Regression and Random Forest.
- Tune the probability threshold based on business requirements.
- Explore class weights or resampling to improve churn-class recall.
- Use cross-validation and systematic hyperparameter tuning.
