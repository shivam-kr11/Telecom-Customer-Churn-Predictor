#=======================
#       Import Libraries
#=======================
from datetime import datetime, timezone  # Adds the current time to saved predictions.
import json  # Reads and writes prediction history as a JSON file.
from pathlib import Path  # Builds file paths that work across operating systems.
import secrets  # Creates unpredictable customer reference IDs.
import string  # Provides the letters and numbers used in those IDs.

import pandas as pd  # Displays prediction history in tables and charts.
import requests  # Sends prediction requests to the FastAPI service.
import streamlit as st  # Builds the interactive web interface.
from login_page import load_app_settings, show_login, show_logout_button


#=======================
#       Initialize App
#=======================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #e8f1ff 0%, #f5f8fc 42%, #edf2f7 100%);
    }
    html, body, [class*="st-"] {font-family: 'DM Sans', 'Segoe UI', sans-serif;}
    .block-container {max-width: 1500px; padding-top: 1rem; padding-bottom: 1.5rem;}
    [data-testid="stMetric"] {position: relative; isolation: isolate;
        background: radial-gradient(circle at 100% 0%, rgba(82,145,230,.14), transparent 48%), #fff;
        border: 1px solid transparent; background-clip: padding-box;
        padding: 1rem 1.1rem; border-radius: 14px;
        box-shadow: 0 5px 16px rgba(31,55,86,.06);
        transition: transform .18s ease, box-shadow .18s ease;}
    [data-testid="stMetric"]::before {content: ""; position: absolute; inset: -1px;
        z-index: -1; border-radius: inherit; padding: 1px;
        background: radial-gradient(circle at 100% 0%, #80aee8, #dfe7f1 55%, #edf2f8);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;}
    [data-testid="stMetric"]:hover {transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(31,55,86,.14);}
    [data-testid="stMetric"] label {color: #60738d !important; font-weight: 600;}
    [data-testid="stMetricValue"] {color: #1f4778; font-weight: 750;}
    [data-testid="stVegaLiteChart"] {background: rgba(255,255,255,.82);
        border: 1px solid #dfe7f1; border-radius: 22px; overflow: hidden;
        padding: .75rem; box-shadow: 0 8px 22px rgba(31,55,86,.08);}
    div[data-testid="stForm"] {background: rgba(255,255,255,.76); border: 1px solid #dfe7f1;
        border-radius: 14px; padding: 1rem 1.2rem;}
    .app-eyebrow {color: #546987; text-transform: uppercase; letter-spacing: .12em;
        font-size: .75rem; font-weight: 700; margin-bottom: .15rem; text-align: center;}
    .block-container h1 {font-family: 'Manrope', 'Segoe UI', sans-serif; text-align: center;
        font-size: clamp(2rem, 3.2vw, 2.8rem); font-weight: 800; letter-spacing: -.045em;
        color: #172b4d; margin: 0 0 .7rem 0; padding: 0; line-height: 1.15;}
    .block-container h2, .block-container h3 {font-family: 'Manrope', 'Segoe UI', sans-serif;
        color: #203653; letter-spacing: -.025em;}
    .block-container p, .block-container label {color: #42546b;}
    [data-testid="stDataFrame"] {width: 100%; border: 1px solid #dfe7f1;
        border-radius: 10px; overflow: hidden;}
    div[data-testid="stDownloadButton"] {width: 100%;}
    div[data-testid="stDownloadButton"] button {width: 100%; min-height: 2.8rem;
        justify-content: center; border-radius: 9px;}
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #fff; border: 1px solid #aebdd0; border-radius: 9px;
        box-shadow: 0 1px 2px rgba(31,55,86,.04);}
    div[data-testid="stNumberInput"] [data-baseweb="input"] {
        background: #fff; border: 1px solid #aebdd0; border-radius: 9px;
        overflow: hidden; box-shadow: 0 1px 2px rgba(31,55,86,.04);}
    div[data-testid="stNumberInput"] input {background: transparent; border: 0; box-shadow: none;}
    div[data-testid="stNumberInput"] button {background: #fff; border: 0;}
    div[data-testid="stTextInput"] input:hover,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
        border-color: #6886ac;}
    div[data-testid="stNumberInput"] [data-baseweb="input"]:hover {border-color: #6886ac;}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: #3574c8; box-shadow: 0 0 0 2px rgba(53,116,200,.18);}
    div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
        border-color: #3574c8; box-shadow: 0 0 0 2px rgba(53,116,200,.18);}
    .block-container > div > div > div > div > p {text-align: center;}
    div[data-testid="stTabs"] {margin-top: .25rem; width: calc(100% - 3.5rem);}
    div[data-testid="stTabs"] > div:first-child {width: 100%;}
    div[data-testid="stTabs"] [role="tablist"] {display: flex; width: 100%; min-width: 0; gap: .45rem;
        border-bottom: 1px solid #d8e1ec;}
    div[data-testid="stTabs"] button[role="tab"] {flex: 1 1 0; justify-content: center;
        min-height: 3.1rem; padding: .55rem .8rem; border-radius: 10px 10px 0 0;
        min-width: 0; white-space: normal; line-height: 1.2; background: rgba(255,255,255,.6);
        color: #49617e; position: relative; overflow: hidden;
        transition: transform .18s ease, box-shadow .18s ease;}
    div[data-testid="stTabs"] button[role="tab"]::after {content: ""; position: absolute;
        left: 16%; right: 16%; bottom: 0; height: 3px; border-radius: 3px 3px 0 0;
        background: #3574c8; transform: scaleX(0); transform-origin: center;
        transition: transform .2s ease;}
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background: rgba(255,255,255,.6); color: #49617e; transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(31,55,86,.1);}
    div[data-testid="stTabs"] button[role="tab"]:hover::after {transform: scaleX(1);}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #fff; border-bottom: 3px solid #3574c8; font-weight: 700;}
    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] {
        background: #2457a6; color: #fff; border: 1px solid #2457a6;
        border-radius: 9px; font-weight: 700; transition: background .15s ease;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover,
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #193f7b; border-color: #193f7b; color: #fff;
    }
    div[data-testid="stForm"] button[kind="primary"]:focus,
    div[data-testid="stButton"] button[kind="primary"]:focus {
        box-shadow: 0 0 0 .2rem rgba(36,87,166,.22); color: #fff;
    }
    @media (max-width: 700px) {
        div[data-testid="stTabs"] {width: 100%;}
        div[data-testid="stTabs"] [role="tablist"] {flex-wrap: wrap;}
        div[data-testid="stTabs"] button[role="tab"] {flex: 1 1 45%;}
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="app-eyebrow">Customer retention workspace</div>', unsafe_allow_html=True)
st.title("Customer churn prediction")

app_settings = load_app_settings()
API_BASE_URL = app_settings["API_BASE_URL"]
API_ACCESS_KEY = app_settings["API_ACCESS_KEY"]

# Stop here until the user has signed in successfully.
if not show_login(app_settings):
    st.stop()

show_logout_button()


#=======================
#       Helper Functions
#=======================
@st.cache_data(ttl=10)
def get_model_info():
    response = requests.get(
        f"{API_BASE_URL}/model-info",
        headers={"X-API-Key": API_ACCESS_KEY},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def load_history():
    try:
        response = requests.get(
            f"{API_BASE_URL}/history",
            headers={"X-API-Key": API_ACCESS_KEY},
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("records", [])
    except requests.RequestException as error:
        st.error(f"Could not load saved predictions from the database: {error}")
        return []


def send_history_request(method, url, **request_options):
    """Send a database history request through the protected API."""
    response = requests.request(
        method,
        f"{API_BASE_URL}{url}",
        headers={"X-API-Key": API_ACCESS_KEY},
        timeout=15,
        **request_options,
    )
    response.raise_for_status()
    return response


def get_category_options(model_info, column):
    """Return categories learned by the fitted one-hot encoder."""
    options = model_info.get("category_options", {}).get(column, [])
    if not options:
        return ["Unknown"]
    return options


def make_customer_id():
    """Create a short random ID for referring to a saved prediction."""
    characters = string.ascii_uppercase + string.digits
    return "CUS-" + "".join(secrets.choice(characters) for _ in range(8))


def make_history_frame(history):
    rows = []
    for record in history:
        row = {
            "Record": record.get("id"),
            "Customer ID": record.get("customer_id", ""),
            "Name": record.get("name", ""),
            "Email": record.get("email", ""),
            "Age": record.get("age", ""),
            "Date and time": record.get("created_at", ""),
            "Result": record.get("predicted_label", ""),
            "Churn probability": record.get("probability_churn", ""),
        }
        features = record.get("features", {})
        for feature_name, feature_value in features.items():
            row[feature_name] = feature_value
        rows.append(row)
    return pd.DataFrame(rows)


#=======================
#       Connect to Model
#=======================
try:
    model_info = get_model_info()
except requests.RequestException as error:
    st.error("The prediction API is not available. Start it with `python -m uvicorn app:app --reload`.")
    st.caption(f"Connection detail: {error}")
    st.stop()

numeric_columns = model_info["numeric_columns"]
categorical_columns = model_info["categorical_columns"]
history = load_history()


def show_dashboard():
    st.subheader("Overview")
    st.caption("A quick look at predictions saved on this device.")
    total_predictions = len(history)
    churned_predictions = sum(record.get("prediction") == 1 for record in history)
    retained_predictions = total_predictions - churned_predictions
    dashboard_columns = st.columns(3)
    dashboard_columns[0].metric("Total predictions", total_predictions)
    dashboard_columns[1].metric("Predicted churn", churned_predictions)
    dashboard_columns[2].metric("Predicted retained", retained_predictions)
    st.caption(f"Active model: **{model_info['model']}** · Decision threshold: **{model_info['threshold']:.2f}**")
    if history:
        chart_data = pd.DataFrame(
            {
                "Predicted outcome": ["Churn", "Retained"],
                "Customers": [churned_predictions, retained_predictions],
            }
        ).set_index("Predicted outcome")
        st.markdown("#### Prediction mix")
        st.bar_chart(chart_data, color="#3976bd", height=260)
        st.markdown("#### Recent predictions")
        recent_predictions = make_history_frame(history).head(10)
        st.dataframe(
            recent_predictions[["Customer ID", "Name", "Date and time", "Result", "Churn probability"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions have been saved yet.")


#=======================
#       App Pages
#=======================
dashboard_tab, prediction_tab, history_tab, insights_tab = st.tabs(
    ["Dashboard", "Prediction", "Prediction History", "Model Insights"]
)

with dashboard_tab:
    show_dashboard()

with prediction_tab:
    st.subheader("Assess a customer")
    st.caption("Fill in the account details below. The model will estimate the likelihood of churn.")
    if "current_customer_id" not in st.session_state:
        st.session_state["current_customer_id"] = make_customer_id()
    id_column, refresh_column = st.columns([4, 1])
    id_column.text_input("Customer ID", value=st.session_state["current_customer_id"], disabled=True)
    refresh_column.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
    if refresh_column.button("New ID", use_container_width=True):
        st.session_state["current_customer_id"] = make_customer_id()
        st.rerun()
    with st.form("prediction_form"):
        features = {}
        input_columns = numeric_columns + categorical_columns
        columns = st.columns(2)
        st.markdown("#### Personal details (optional)")
        personal_columns = st.columns(3)
        customer_name = personal_columns[0].text_input("Name")
        customer_email = personal_columns[1].text_input("Email")
        customer_age = personal_columns[2].number_input("Age", min_value=0, max_value=120, value=None, step=1, placeholder="Optional")
        st.markdown("#### Account details")
        for index, column in enumerate(input_columns):
            with columns[index % 2]:
                if column in numeric_columns:
                    if column == "Tenure Months":
                        default_value = 12.0
                    elif column == "Monthly Charges":
                        default_value = 70.0
                    elif column == "Total Charges":
                        default_value = 800.0
                    elif column == "CLTV":
                        default_value = 4000.0
                    else:
                        default_value = 0.0
                    step_value = 1.0 if column == "Tenure Months" else 0.1
                    features[column] = st.number_input(
                        column,
                        value=default_value,
                        step=step_value,
                        format="%.2f",
                    )
                else:
                    options = get_category_options(model_info, column)
                    features[column] = st.selectbox(column, options)
        submitted = st.form_submit_button("Check prediction", type="primary", use_container_width=True)

    if submitted:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": features},
                headers={"X-API-Key": API_ACCESS_KEY},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            st.session_state["last_prediction"] = result
            record = {
                "customer_id": st.session_state["current_customer_id"],
                "name": customer_name.strip(),
                "email": customer_email.strip(),
                "age": customer_age,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prediction": result["prediction"],
                "predicted_label": result["predicted_label"],
                "probability_churn": result["probability_churn"],
                "threshold": result["threshold"],
                "features": features,
            }
            send_history_request("POST", "/history", json=record)
            st.success("Estimate complete. This prediction was saved to the database.")
        except requests.RequestException as error:
            if isinstance(error, requests.HTTPError) and error.response is not None:
                try:
                    detail = error.response.json().get("detail", str(error))
                except ValueError:
                    detail = str(error)
                st.error(f"Prediction failed: {detail}")
            else:
                st.error(f"Prediction failed: {error}")

    result = st.session_state.get("last_prediction")
    if result:
        st.markdown("### Prediction result")
        result_columns = st.columns(3)
        result_columns[0].metric("Prediction", result["predicted_label"])
        result_columns[1].metric("Churn probability", f"{result['probability_churn']:.1%}")
        result_columns[2].metric("Decision threshold", f"{result['threshold']:.2f}")
        if result["prediction"] == 1:
            st.warning("Higher estimated churn risk: the probability is at or above the model threshold.")
        else:
            st.info("Lower estimated churn risk: the probability is below the model threshold.")
        st.progress(min(max(float(result["probability_churn"]), 0.0), 1.0), text="Estimated churn probability")
        st.caption("This is a model estimate based on the supplied account details, not a certainty about what the customer will do.")

with history_tab:
    st.subheader("Saved predictions")
    st.caption("Search, filter, export, or remove predictions stored on this device.")
    history = load_history()
    if history:
        history_frame = make_history_frame(history)
        label_filter = st.selectbox(
            "Filter by prediction",
            ["All", "Churned", "Retained"],
        )
        search_text = st.text_input("Search saved features or dates")
        filtered_frame = history_frame.copy()
        if label_filter != "All":
            filtered_frame = filtered_frame[filtered_frame["Result"] == label_filter]
        if search_text.strip():
            search_mask = filtered_frame.astype(str).apply(
                lambda column: column.str.contains(search_text.strip(), case=False, regex=False)
            ).any(axis=1)
            filtered_frame = filtered_frame[search_mask]
        st.dataframe(
            filtered_frame[["Customer ID", "Name", "Email", "Age", "Date and time", "Result", "Churn probability"]],
            use_container_width=True,
            hide_index=True,
        )
        download_columns = st.columns(2, gap="small")
        with download_columns[0]:
            st.download_button(
                "Download filtered results (CSV)",
                filtered_frame.drop(columns=["Record"]).to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with download_columns[1]:
            st.download_button(
                "Download all history (JSON)",
                json.dumps(history, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="prediction_history.json",
                mime="application/json",
                use_container_width=True,
            )

        record_numbers = filtered_frame["Record"].tolist()
        labels = {
            row["Record"]: f"{row['Date and time']} — {row['Result']}"
            for _, row in filtered_frame.iterrows()
        }
        selected_records = st.multiselect(
            "Select prediction records to delete",
            options=record_numbers,
            format_func=lambda record_number: labels.get(record_number, str(record_number)),
        )
        if selected_records and st.button("Delete selected records"):
            st.session_state["confirm_delete_records"] = selected_records
        if st.session_state.get("confirm_delete_records"):
            st.warning("Confirm deletion of the selected prediction records.")
            confirm_columns = st.columns(2)
            if confirm_columns[0].button("Confirm deletion", type="primary"):
                send_history_request(
                    "POST",
                    "/history/delete-selected",
                    json=st.session_state["confirm_delete_records"],
                )
                st.session_state.pop("confirm_delete_records", None)
                st.rerun()
            if confirm_columns[1].button("Cancel deletion"):
                st.session_state.pop("confirm_delete_records", None)
                st.rerun()

        st.markdown("#### Prediction details")
        for _, row in filtered_frame.iterrows():
            with st.expander(f"{row['Customer ID']} · {row['Result']} · {row['Date and time']}"):
                st.write({
                    "Name": row.get("Name", ""),
                    "Email": row.get("Email", ""),
                    "Age": row.get("Age", ""),
                    "Churn probability": row.get("Churn probability", ""),
                })
                feature_details = {}
                for column in numeric_columns + categorical_columns:
                    if column in row.index:
                        feature_details[column] = row[column]
                st.write("Account details", feature_details)

        st.markdown("#### Remove saved data")
        st.caption("This also deletes saved names, email addresses, ages, and account inputs from this device.")
        if st.button("Delete all prediction history"):
            st.session_state["confirm_delete_all"] = True
        if st.session_state.get("confirm_delete_all"):
            st.warning("This permanently removes all prediction records from the database.")
            clear_columns = st.columns(2)
            if clear_columns[0].button("Confirm: delete all", type="primary"):
                send_history_request("DELETE", "/history")
                st.session_state.pop("confirm_delete_all", None)
                st.rerun()
            if clear_columns[1].button("Cancel", key="cancel_delete_all"):
                st.session_state.pop("confirm_delete_all", None)
                st.rerun()
    else:
        st.info("No predictions have been saved yet.")

with insights_tab:
    st.subheader("About this model")
    st.caption("Details about the model currently serving predictions.")
    insight_columns = st.columns(3)
    insight_columns[0].metric("Selected model", model_info["model"])
    insight_columns[1].metric("Decision threshold", f"{model_info['threshold']:.2f}")
    insight_columns[2].metric("Model input fields", model_info["selected_input_count"])
    st.write("Inputs used by the model:")
    st.write(model_info["selected_input_columns"])
    st.caption(
        "Prediction history is stored in the connected MySQL database. "
        "The model returns a churn probability; the threshold converts it to a class label."
    )


