<<<<<<< HEAD
# Telecom-Customer-Churn-Predictor
A machine learning project that predicts telecom customer churn using five classifiers, with a FastAPI prediction service and Streamlit dashboard.
=======
# Customer Churn Prediction

This project uses customer account information to predict whether a telecom customer may leave the service. It includes a training script, a small prediction API, and a Streamlit interface for trying predictions.

## Dataset

The project uses the Telco Customer Churn workbook at `data/raw/Telco_customer_churn.xlsx` (sheet: `Telco_Churn`). The `Churn Value` column is the outcome: `1` means the customer churned and `0` means they stayed.

Before training, the script checks the customer IDs and churn labels, converts `Total Charges` to numbers, and removes columns that duplicate or reveal the outcome (`Churn Label`, `Churn Score`, and `Churn Reason`). Missing numeric feature values are filled with medians calculated from the training data. Possible outliers are reported and plotted, but kept because they may represent real customers.

## Project files

- `main.py` — explores and prepares the data, trains and compares models, saves charts and the selected model
- `app.py` — serves predictions through a FastAPI API
- `streamlit_app.py` — provides a form for predictions, prediction history, and model insights
- `data/raw/` — source workbook
- `data/processed/` — cleaned data created by the training workflow
- `data/submissions/` — local prediction history created by Streamlit
- `models/` — saved model and preprocessing information
- `reports/model_comparison.csv` — accuracy, precision, recall, and F1 results
- `reports/figures/` — charts created by the training workflow
- `requirements.txt` — Python packages needed for the project

## Set up

Install Python 3.10 or newer. Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the models

Run the training script from the project folder:

```powershell
python main.py
```

This creates the model comparison report, charts, and saved model. The script trains Logistic Regression, K-Nearest Neighbors, Naive Bayes, Decision Tree, and AdaBoost. For K-Nearest Neighbors, it tries values of K from 1 to 20 and chooses the one with the best F1 score on a holdout portion of the training data. It chooses the saved model using a validation split from the training data. The separate test split is used for the final comparison table, which reports accuracy, precision, recall, and F1 score.

## Run the prediction app

Start the API in one PowerShell window:

```powershell
python -m uvicorn app:app --reload
```

Then start Streamlit in a second window:

```powershell
streamlit run streamlit_app.py
```

Streamlit sends prediction requests to the local API at `http://127.0.0.1:8000`.

## How to read the results

The data is split into training and test groups while keeping the churn proportions similar. Preprocessing steps are learned from the training group and then applied to the test group. Customer IDs, location details, and outcome-related columns are left out of the model inputs.

Accuracy is the share of all predictions that were correct. Precision tells you how often a predicted churn was actually churn. Recall tells you how many actual churn cases the model found. F1 combines precision and recall into one score. These results come from the project’s test split and should be treated as a comparison of these models, not a guarantee of future performance.

Streamlit stores prediction details locally in `data/submissions/prediction_inputs.json`, including optional name, email, and age fields. The Prediction History page can delete individual records or clear all saved history. Treat this file as sensitive and use the API only on a trusted machine; authentication is not implemented.
>>>>>>> b3124d8 (Project Done up to Ml model training, ui, fast api impleneted)
