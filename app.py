# Import the tools used by the API.
from pathlib import Path  # Helps find files using paths relative to this script.
from typing import Any  # Allows the request to contain values of different types.

import joblib  # Loads the trained model and its saved preprocessing tools.
import pandas as pd  # Organizes customer inputs into tables for the model.
from fastapi import FastAPI, Header, HTTPException  # Builds routes, reads headers, and returns errors.
from pydantic import BaseModel  # Checks the structure of incoming API requests.
from auth import get_setting, api_key_matches
import database  # Saves and retrieves prediction history in MySQL.

# Create the FastAPI application. FastAPI automatically provides API docs at /docs.
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for customer classification using the selected churn model.",
    version="1.0.0",
)

# The project folder is where this file lives. This keeps file loading reliable
# even when the API is started from a different folder.
# Find the saved model file next to this script, regardless of the launch folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_PATH}. Run main.py before starting the API."
    )

artifacts = joblib.load(MODEL_PATH)

# Load the trained model and the preprocessing tools saved by main.py.
# The imputer fills missing numbers, the encoders prepare text choices, and
# the scaler puts inputs on the same numeric scale used during training.
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
API_ACCESS_KEY = get_setting("API_ACCESS_KEY")


def check_api_key(api_key: str | None):
    """Allow private API routes only when the shared key is correct."""
    if not API_ACCESS_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured. Set API_ACCESS_KEY before starting the service.",
        )
    if api_key is None or not api_key_matches(api_key, API_ACCESS_KEY):
        raise HTTPException(status_code=401, detail="Login required or API key is invalid.")


@app.get("/")
def home():
    """Show a short message confirming that the API is running."""
    # A browser can open this address to check that the API started correctly.
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
    }


@app.get("/model-info")
def model_info(api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Return details the Streamlit form needs to build its input fields."""
    check_api_key(api_key)
    # Streamlit uses these names and choices to display the correct form fields.
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


class PredictionRequest(BaseModel):
    """Describe the JSON format accepted by the prediction endpoint."""
    # The client sends one dictionary of feature names and values.
    features: dict[str, Any]


@app.post("/predict")
def predict_customer(
    request: PredictionRequest,
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Prepare one customer's details and ask the model for a churn estimate."""
    check_api_key(api_key)
    required_columns = numerical_columns + categorical_columns

    # Stop early if the request left out any model input fields.
    missing_columns = []
    for column in required_columns:
        if column not in request.features:
            missing_columns.append(column)

    if missing_columns:
        # HTTP 422 means the request needs to be corrected by the caller.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features",
                "missing_features": missing_columns,
            },
        )

    # Check that text choices are among the values used to train the model.
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
        # Prevent unseen text choices from silently becoming misleading inputs.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unsupported category value. Choose a value used during training.",
                "invalid_categories": unsupported_values,
            },
        )

    # Make a one-row table containing the customer's submitted information.
    feature_row = {}
    for column in required_columns:
        feature_row[column] = request.features[column]
    raw_features = pd.DataFrame([feature_row])

    # Convert numbers to numeric values. Invalid or blank values become missing.
    for column in numerical_columns:
        raw_features[column] = pd.to_numeric(raw_features[column], errors="coerce")

    # Fill missing numbers with the same medians used during training.
    numeric_values = numeric_imputer.transform(raw_features[numerical_columns])
    numeric_data = pd.DataFrame(numeric_values, columns=numerical_columns)

    # Clean up category text before encoding it for the model.
    for column in categorical_columns:
        raw_features[column] = raw_features[column].fillna("Unknown").astype(str).str.strip()

    # Gender uses the label encoder fitted during training.
    gender_data = pd.DataFrame(index=[0])
    if "Gender" in categorical_columns:
        try:
            gender_data["Gender"] = label_encoder.transform(raw_features["Gender"])
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail="Gender must match one of the categories learned during training.",
            ) from error

    # Turn the other text choices into 0/1 columns, just like training did.
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
        # Use an empty one-row table if there are no other category fields.
        encoded_categories = pd.DataFrame(index=[0])

    # Join all prepared fields and put them in the exact order expected by the model.
    combined = pd.concat([numeric_data, gender_data, encoded_categories], axis=1)
    combined = combined.reindex(columns=feature_columns, fill_value=0)

    # Apply the saved scaler, then ask the model for the churn probability.
    scaled_features = pd.DataFrame(
        scaler.transform(combined),
        columns=feature_columns,
    )
    probability = float(model.predict_proba(scaled_features)[0, 1])
    prediction = int(probability >= threshold)

    # Return both the probability and the final class decision.
    # A probability at or above the threshold is labeled as churn.
    return {
        "prediction": prediction,
        "predicted_label": "Churned" if prediction == 1 else "Retained",
        "probability_churn": round(probability, 4),
        "threshold": threshold,
    }


@app.get("/history")
def get_prediction_history(
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Return saved predictions for the history and dashboard pages."""
    check_api_key(api_key)
    return {"records": database.get_predictions()}


@app.post("/history")
def save_prediction_history(
    record: dict[str, Any],
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Save one prediction to MySQL."""
    check_api_key(api_key)
    return database.save_prediction(record)


@app.delete("/history/{record_id}")
def remove_prediction_history(
    record_id: int,
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Delete one prediction using its database ID."""
    check_api_key(api_key)
    database.delete_prediction(record_id)
    return {"message": "Prediction deleted."}


@app.delete("/history")
def clear_prediction_history(
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Delete all saved prediction history."""
    check_api_key(api_key)
    database.delete_predictions()
    return {"message": "Prediction history cleared."}


@app.post("/history/delete-selected")
def delete_selected_prediction_history(
    record_ids: list[int],
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Delete several selected history rows."""
    check_api_key(api_key)
    database.delete_predictions(record_ids)
    return {"message": "Selected predictions deleted."}



