---
title: Medical Insurance Cost Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

## 🏥 About

This app uses a Gradient Boosting Regressor trained on the Kaggle Medical Cost
Personal Dataset to predict an individual's annual medical insurance charges.
It is designed to help users understand how personal health and demographic
factors influence insurance costs, presented through an interactive Gradio web
interface.

## 📊 Dataset

- **Source:** Kaggle Medical Cost Personal Dataset
- **Size:** 1,338 patient records, 6 input features, 1 target variable (`charges`)
- **Link:** https://www.kaggle.com/datasets/mirichoi0218/insurance

## 🔢 Input Features

| Feature  | Type    | Description                          | Range / Values                                      |
|----------|---------|--------------------------------------|-----------------------------------------------------|
| age      | Integer | Age of the primary beneficiary       | 18 – 64                                             |
| sex      | String  | Biological sex of the policyholder   | male / female                                       |
| bmi      | Float   | Body Mass Index                      | 10.0 – 55.0  (normal: 18.5 – 24.9)                 |
| children | Integer | Number of children / dependents      | 0 – 5                                               |
| smoker   | String  | Whether the beneficiary smokes       | yes / no                                            |
| region   | String  | Residential area in the US           | northeast / northwest / southeast / southwest       |

## 🤖 Model Performance

| Model              | RMSE ($) | MAE ($) | R² Score |
|--------------------|----------|---------|----------|
| LinearRegression   |          |         |          |
| Ridge              |          |         |          |
| Lasso              |          |         |          |
| DecisionTree       |          |         |          |
| RandomForest       |          |         |          |
| GradientBoosting   |          |         |          |

> Run `python train.py` and copy the printed table values into the rows above.

## 🚀 How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `insurance.csv` in the project root (`insurance-predictor/`).
3. Train the model and generate all artifacts:
   ```bash
   python train.py
   ```
4. Launch the Gradio web app:
   ```bash
   python app.py
   ```
5. Open the local URL printed in your terminal (e.g., `http://127.0.0.1:7860`).

## ⚠️ Disclaimer

This application is for **educational purposes only** and does not constitute
financial or medical advice. Predictions are based on statistical patterns in
historical data and should not be used to make real insurance or health decisions.
