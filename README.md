# Telecom Customer Churn Predictor

A beginner-friendly machine learning project that estimates whether a telecom customer may leave the service. It includes a model training workflow, a FastAPI prediction service, and a Streamlit dashboard.

## What the project does

- Checks and prepares the Telco Customer Churn dataset.
- Trains and compares five classification models.
- Provides an API and a simple web interface for predictions.
- Saves prediction history in MySQL so it can be reviewed, downloaded, or deleted.

## Dataset

The project reads `data/raw/Telco_customer_churn.xlsx`, using the `Telco_Churn` worksheet. The target column is `Churn Value`: `1` means the customer churned and `0` means they stayed.

During preparation, the training script checks customer IDs and verifies that `Churn Label` agrees with `Churn Value`. It converts `Total Charges` to a number, fills missing numeric inputs using medians from the training split, and reports possible outliers. Outliers are kept because they may be valid customer records. Columns that duplicate or reveal the outcome, including `Churn Label`, `Churn Score`, and `Churn Reason`, are excluded from model inputs.

## Models and current results

The project compares Logistic Regression, AdaBoost, Naive Bayes, K-Nearest Neighbors, and Decision Tree. The table below comes from the current test split; values are rounded to one decimal place.

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.5% | 65.0% | 57.5% | 61.0% |
| AdaBoost | 80.5% | 65.1% | 57.0% | 60.8% |
| Naive Bayes | 72.2% | 48.5% | 80.5% | 60.6% |
| K-Nearest Neighbors | 74.7% | 52.2% | 52.9% | 52.6% |
| Decision Tree | 71.8% | 47.0% | 48.4% | 47.7% |

The saved model is selected by F1 score on a validation split taken from the training data. The separate test split is used for the comparison table. These results describe this dataset split and are not a guarantee of performance on future customers.

## Project structure

```text
├── app.py                         # FastAPI prediction service
├── auth.py                        # Reads private settings and checks credentials
├── database.py                    # Creates and manages MySQL prediction history
├── login_page.py                  # Displays the sign-in screen
├── main.py                        # Data preparation, training, evaluation, and export
├── streamlit_app.py               # Streamlit dashboard and prediction form
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── secrets.toml.example       # Template for local private settings
├── data/
│   ├── raw/                        # Source Excel workbook
│   ├── processed/                  # Generated cleaned data
├── models/                         # Saved model and preprocessing tools
└── reports/
    ├── model_comparison.csv        # Model evaluation metrics
    └── figures/                    # Generated charts
```

## Set up

Install Python 3.11 or newer. From the project folder, create and activate a virtual environment, then install the dependencies:

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

The script writes the model comparison to `reports/model_comparison.csv`, refreshes the charts in `reports/figures/`, and saves the selected model and preprocessing objects in `models/churn_model.pkl`. For K-Nearest Neighbors, it checks K values from 1 to 20 and selects one using validation F1.

## Start the prediction app

Authentication helpers are in `auth.py`, and the sign-in screen is in `login_page.py`. Prediction history is handled by `database.py` and saved in MySQL. Create a local settings file once by copying `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then replace the example values with your MySQL Workbench connection details and login secrets. The real `secrets.toml` file is ignored by Git, so it won’t be committed.

```powershell
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Use the same `API_ACCESS_KEY` for the local API and Streamlit app. Set `DB_USER` and `DB_PASSWORD` to the MySQL account you use with Workbench. `DB_NAME` should match the schema you created, `customer_churn_db`. The app creates its `prediction_history` table the first time it saves or loads history. You only need to edit the private settings file once; the values remain available when you open new terminal windows.

Start the API in one terminal:

```powershell
python -m uvicorn app:app --reload
```

In a second terminal, start the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

The Streamlit app connects to the API at `http://127.0.0.1:8000`. The API documentation is available at `http://127.0.0.1:8000/docs` while the API is running.

## Deploy

The web interface and prediction API are separate services. The repository includes a `render.yaml` blueprint for deploying the FastAPI service to Render. In Render, configure `API_ACCESS_KEY` and the `DB_*` settings. The API needs a MySQL server that Render can reach; your local MySQL Workbench server is only reachable from your computer, so it cannot serve the deployed API. The saved model at `models/churn_model.pkl` must be included in the repository.

Deploy the Streamlit interface separately on [Streamlit Community Cloud](https://share.streamlit.io/), selecting this repository and `streamlit_app.py` as the app file. In the app's **Settings → Secrets**, add the public API URL, the shared login, and the same API key configured in Render:

```toml
API_BASE_URL = "https://your-api-service.onrender.com"
APP_USERNAME = "your-login-name"
APP_PASSWORD = "use-a-strong-password"
API_ACCESS_KEY = "the-same-long-random-secret-set-in-render"
```

The app also reads these values from environment variables. For local use, `API_BASE_URL` defaults to `http://127.0.0.1:8000`. Do not include a trailing slash in the configured URL (the app removes it automatically). This project uses one shared login; it does not provide separate accounts or password reset.

The Streamlit interface sends history requests to the API. The API reads and writes those records in the configured MySQL database.

## Understanding the metrics

- **Accuracy:** the share of all predictions that were correct.
- **Precision:** among customers predicted to churn, the share who did churn.
- **Recall:** among customers who churned, the share the model found.
- **F1 score:** a combined measure of precision and recall.

High recall can help find more customers at risk, while high precision means fewer false alarms. Which metric matters most depends on how the business plans to use the predictions.

## Local prediction history and privacy

Prediction history is stored in the MySQL `prediction_history` table. Records can include the generated customer reference ID, account inputs, and optional name, email, and age. The history page lets you remove selected records or delete all saved records.

The prediction and model information API routes require the shared API key. The root health message remains public. The Streamlit app displays login and prediction history on its host, so use strong credentials and avoid entering real customer personal information in a public demo deployment.
