#=======================
#       Import Libraries
#=======================
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


#=======================
#       Initialize API
#=======================
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for customer classification using the selected churn model.",
    version="1.0.0",
)


#=======================
#       Load Model Artifacts
#=======================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_PATH}. Run main.py before starting the API."
    )

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
numeric_imputer = artifacts["numeric_imputer"]
label_encoder = artifacts["label_encoder"]
scaler = artifacts["scaler"]
numerical_columns = artifacts["numerical_columns"]
categorical_columns = artifacts["categorical_columns"]
other_categorical_columns = artifacts["other_categorical_columns"]
dummy_columns = artifacts["dummy_columns"]
feature_columns = artifacts["feature_columns"]
selected_input_columns = artifacts["selected_input_columns"]
category_options = artifacts["category_options"]
threshold = float(artifacts["threshold"])


#=======================
#       Home Endpoint
#=======================
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
    }


#=======================
#       Model Information
#=======================
@app.get("/model-info")
def model_info():
    return {
        "model": artifacts.get("model_name", type(model).__name__),
        "threshold": threshold,
        "numeric_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "category_options": category_options,
        "selected_input_count": len(selected_input_columns),
        "selected_input_columns": selected_input_columns,
        "total_feature_count": len(feature_columns),
    }


#=======================
#       Prediction Input Schema
#=======================
class PredictionRequest(BaseModel):
    features: dict[str, Any]


#=======================
#       Prediction Endpoint
#=======================
@app.post("/predict")
def predict_customer(request: PredictionRequest):
    required_columns = numerical_columns + categorical_columns
    missing_columns = [
        column for column in required_columns if column not in request.features
    ]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features",
                "missing_features": missing_columns,
            },
        )

    unsupported_values = {}
    for column in categorical_columns:
        value = str(request.features[column]).strip()
        allowed_values = category_options.get(column, [])
        if allowed_values and value not in allowed_values:
            unsupported_values[column] = {
                "received": value,
                "allowed": allowed_values,
            }
    if unsupported_values:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unsupported category value. Choose a value used during training.",
                "invalid_categories": unsupported_values,
            },
        )

    raw_features = pd.DataFrame(
        [{column: request.features[column] for column in required_columns}]
    )
    for column in numerical_columns:
        raw_features[column] = pd.to_numeric(raw_features[column], errors="coerce")

    numeric_values = numeric_imputer.transform(raw_features[numerical_columns])
    numeric_data = pd.DataFrame(numeric_values, columns=numerical_columns)

    for column in categorical_columns:
        raw_features[column] = raw_features[column].fillna("Unknown").astype(str).str.strip()

    gender_data = pd.DataFrame(index=[0])
    if "Gender" in categorical_columns:
        try:
            gender_data["Gender"] = label_encoder.transform(raw_features["Gender"])
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail="Gender must match one of the categories learned during training.",
            ) from error

    if other_categorical_columns:
        encoded_categories = pd.get_dummies(
            raw_features[other_categorical_columns],
            columns=other_categorical_columns,
            dtype="int",
        )
        encoded_categories = encoded_categories.reindex(
            columns=dummy_columns,
            fill_value=0,
        )
    else:
        encoded_categories = pd.DataFrame(index=[0])

    combined = pd.concat([numeric_data, gender_data, encoded_categories], axis=1)
    combined = combined.reindex(columns=feature_columns, fill_value=0)
    scaled_features = pd.DataFrame(
        scaler.transform(combined),
        columns=feature_columns,
    )
    probability = float(model.predict_proba(scaled_features)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "predicted_label": "Churned" if prediction == 1 else "Retained",
        "probability_churn": round(probability, 4),
        "threshold": threshold,
    }



