# Telco Customer Churn Prediction
![Python](https://img.shields.io/badge/Python-3.9-blue)
![H2O.ai](https://img.shields.io/badge/H2O-AutoML-orange)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-lightgrey)

This notebook demonstrates a machine learning workflow to predict customer churn in a telecommunications dataset using H2O's AutoML capabilities. It covers data loading, model training, evaluation, customer segmentation, and visualization.

## 1. Environment Setup

The necessary libraries, `h2o` and `kaggle`, are installed. An H2O cluster is initialized for distributed machine learning tasks.

```python
!pip install h2o
!pip install kaggle
import h2o
from h2o.automl import H2OAutoML
h2o.init()
```

## 2. Data Loading

The Telco Customer Churn dataset is downloaded from Kaggle (requires a Kaggle API key) and then imported into the H2O environment.

```python
!kaggle datasets download -d blastchar/telco-customer-churn -p /content
!unzip /content/telco-customer-churn.zip -d /content
data = h2o.import_file("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

## 3. AutoML Model Training

The `Churn` column is set as the target variable (`y`), and all other columns are used as features (`x`). H2O AutoML is then used to automatically train and tune various machine learning models (e.g., GBM, GLM, Deep Learning, XGBoost) to predict customer churn. It runs for a maximum of 20 models.

```python
y = "Churn"
x = data.columns
x.remove(y)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=data)
```

## 4. Model Evaluation and Prediction

The leaderboard of the trained models is displayed, showing their performance metrics. The best performing model (leader model) is then used to make predictions on the dataset.

```python
lb = aml.leaderboard
print(lb)

model = aml.leader
pred = model.predict(data)
print(pred.head())
```

## 5. Feature Importance

The variable importance plot for one of the Gradient Boosting Machine (GBM) models is displayed to understand which features contribute most to the churn prediction.

```python
gbm_model = h2o.get_model("GBM_grid_1_AutoML_1_20260518_65151_model_2") # Example model ID
gbm_model.varimp_plot()
```

## 6. Customer Segmentation

Customers are segmented into 'High Risk', 'Medium Risk', and 'Low Risk' categories based on their `Contract` type, `tenure`, and `OnlineSecurity` status. Specifically:

*   **High Risk**: Month-to-month contract, tenure less than 12 months, and no online security.
*   **Medium Risk**: Month-to-month contract, tenure 12 months or more.
*   **Low Risk**: All other customers.

```python
cond_high_risk = (data['Contract'] == 'Month-to-month') & (data['tenure'] < 12) & (data['OnlineSecurity'] == 'No')
cond_medium_risk = (data['Contract'] == 'Month-to-month') & (data['tenure'] >= 12)
data['Segment'] = cond_high_risk.ifelse('High Risk', cond_medium_risk.ifelse('Medium Risk', 'Low Risk'))

data['Segment'].as_data_frame().value_counts()
```

## 7. Segment Distribution Visualization

A bar plot is generated to visualize the distribution of customers across the defined risk segments.

```python
import matplotlib.pyplot as plt

segment_counts = data['Segment'].as_data_frame().value_counts()
segment_counts.plot(kind='bar', color=['green','orange','red'])
plt.title("Distribusi Segmen Pelanggan")
plt.xlabel("Segment")
plt.ylabel("Jumlah Pelanggan")
plt.show()
```

This analysis provides insights into customer churn behavior and identifies different risk groups, which can be valuable for targeted retention strategies.
