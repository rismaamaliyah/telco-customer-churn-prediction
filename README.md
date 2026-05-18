# Telco Customer Churn Prediction
![Python](https://img.shields.io/badge/Python-3.9-blue)
![H2O.ai](https://img.shields.io/badge/H2O-AutoML-orange)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-lightgrey)

This project demonstrates a complete machine learning workflow to predict customer churn in a telecommunications dataset using **H2O AutoML**. It covers data preparation, automated model training, evaluation, feature importance analysis, customer segmentation, and visualization. The goal is to provide actionable insights for customer retention strategies.

## 1. Environment Setup

Install required libraries and initialize the `H2O` cluster for distributed machine learning. 

```python
!pip install h2o
import h2o
from h2o.automl import H2OAutoML
h2o.init()
```

## 2. Data Loading

Download the Telco Customer Churn dataset from Kaggle and import it into the H2O environment.

```python
!kaggle datasets download -d blastchar/telco-customer-churn -p /content
!unzip /content/telco-customer-churn.zip -d /content
data = h2o.import_file("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

## 3. AutoML Model Training

Define the target variable (`Churn`) and features, then run H2O AutoML to train and tune multiple models (GBM, GLM, Deep Learning, XGBoost).

```python
y = "Churn"
x = data.columns
x.remove(y)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=data)
```

## 4. Model Evaluation and Prediction

Display the leaderboard and use the best model (Stacked Ensemble, AUC ~0.85) to generate predictions.

```python
lb = aml.leaderboard
print(lb)

model = aml.leader
pred = model.predict(data)
print(pred.head())
```

## 5. Feature Importance

Analyze feature importance from a Gradient Boosting Machine (GBM) model to identify key drivers of churn.

```python
gbm_model = h2o.get_model("GBM_grid_1_AutoML_1_20260518_65151_model_2") # Example model ID
gbm_model.varimp_plot()
```

**Top features:** Contract, Tenure, Online Security

## 6. Customer Segmentation

Segment customers into risk groups based on contract type, tenure, and online security status:

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

Visualize customer distribution across segments.

```python
import matplotlib.pyplot as plt

segment_counts = data['Segment'].as_data_frame().value_counts()
segment_counts.plot(kind='bar', color=['green','orange','red'])
plt.title("Distribusi Segmen Pelanggan")
plt.xlabel("Segment")
plt.ylabel("Jumlah Pelanggan")
plt.show()
```

## 📊 Result & Insights
- **Best Model**: Stacked Ensemble (AUC = 0.85).
- **Key Drivers**: Contract type, tenure, online security.
- **Segmentation**:
  - Low Risk: 3,761 customers
  - Medium Risk: 1,967 customers
  - High Risk: 1,315 customers
 
### Business Takeaways:
- Customers with month-to-month contracts are more likely to churn → target with contract upgrade offers.
- New customers (low tenure) need onboarding and loyalty programs.
- Lack of online security services indicates upsell opportunities.

 ## 🚀 Next Steps
 - Deploy the model for real-time churn scoring.
 - Integrate with BI dashboards for marketing teams.
 - Design retention cammpaigns based on risk segmentation.
