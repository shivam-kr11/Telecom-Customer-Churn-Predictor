# ==============================================================================
# PROJECT OBJECTIVES:
# 1. Explore the IBM Telco Customer Churn dataset and identify useful patterns.
# 2. Check data quality, handle missing values, and review possible outliers.
# 3. Encode and scale customer features for machine learning.
# 4. Train Logistic Regression, KNN, Naive Bayes, Decision Tree, and AdaBoost.
# 5. Compare the models and save the best model for customer churn prediction.
# ==============================================================================

#================ IMPORTING LIBRARIES ================
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#================ LOADING DATASET ================
data = pd.read_excel("data/raw/Telco_customer_churn.xlsx", sheet_name="Telco_Churn")

#================ OBJECTIVE 1: EXPLORE CUSTOMER CHURN DATA ================
#================ BASIC DATASET INFORMATION ================
print(data.head())
print(data.shape)
print(data.columns)

#================ DATASET INFORMATION ================
print(data.info())

#================ STATISTICAL SUMMARY ================
print(data.describe())

#================ CHECKING MISSING VALUES ================
print(data.isnull().sum())
print("Total missing values:", data.isnull().sum())


#================ CHECKING DUPLICATE VALUES ================
print("Total duplicate rows:", data.duplicated().sum())
print("Duplicate customer IDs:", data["CustomerID"].duplicated().sum())

#================ CHECKING UNIQUE VALUES ================
print(data.nunique())

#================ TARGET VARIABLE DISTRIBUTION ================
print(data["Churn Value"].value_counts())
print(data["Churn Value"].value_counts(normalize=True) * 100)

#================ CATEGORICAL COLUMN DISTRIBUTION ================
print(data["Gender"].value_counts())
print(data["Contract"].value_counts())
print(data["Internet Service"].value_counts())
print(data["Payment Method"].value_counts())

#================ OBJECTIVE 2: CHECKING DATA QUALITY ================
#================ VERIFYING REQUIRED COLUMNS ================
required_columns = ["CustomerID", "Churn Label", "Churn Value"]
missing_columns = []
for name in required_columns:
    if name not in data.columns:
        missing_columns.append(name)
if missing_columns:
    raise ValueError(f"Required columns are missing from the dataset: {missing_columns}")

#================ REMOVING DUPLICATE CUSTOMER RECORDS ================
data = data.drop_duplicates(subset="CustomerID").copy()
data["Churn Value"] = pd.to_numeric(data["Churn Value"], errors="coerce")
data = data.dropna(subset=["CustomerID", "Churn Value"])

invalid_targets = ~data["Churn Value"].isin([0, 1])
if invalid_targets.any():
    invalid_values = data.loc[invalid_targets, "Churn Value"].unique().tolist()
    raise ValueError(f"Churn Value must contain only 0 and 1. Found: {invalid_values}")
data["Churn Value"] = data["Churn Value"].astype(int)

#================ VERIFYING TARGET LABEL CONSISTENCY ================
mismatch_count = 0
for row_number in range(len(data)):
    churn_label = data.iloc[row_number]["Churn Label"]
    churn_value = data.iloc[row_number]["Churn Value"]

    if pd.isna(churn_label):
        continue

    churn_label = str(churn_label).strip().lower()
    if churn_value == 0:
        expected_label = "no"
    else:
        expected_label = "yes"

    if churn_label != expected_label:
        mismatch_count += 1

if mismatch_count > 0:
    raise ValueError(
        f"Churn Label and Churn Value disagree for {mismatch_count} rows."
    )

#================ CONVERTING TOTAL CHARGES TO NUMERICAL DATA ================
data["Total Charges"] = pd.to_numeric(
    data["Total Charges"].astype("string").str.strip(),
    errors="coerce",
)

# Churn Label duplicates the target; Churn Score and Churn Reason reveal the outcome.
data = data.drop(
    columns=["Churn Label", "Churn Score", "Churn Reason"],
    errors="ignore",
)
data = data.rename(columns={"Churn Value": "target"})

print("Missing values after cleaning:")
print(data.isnull().sum())
print("Total missing values after cleaning:", data.isnull().sum().sum())
print("Target distribution after cleaning:")
print(data["target"].value_counts())

#================ OBJECTIVE 1: TARGET VARIABLE VISUALIZATION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="target", data=data)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/figures/01_target_distribution.png", dpi=200, bbox_inches="tight")
plt.close()

#================ NUMERICAL FEATURE DISTRIBUTION ================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(data["Tenure Months"], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Tenure Months Distribution")
sns.histplot(data["Monthly Charges"], bins=20, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Monthly Charges Distribution")
sns.histplot(data["Total Charges"].dropna(), bins=20, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Total Charges Distribution")
sns.histplot(data["CLTV"], bins=20, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Customer Lifetime Value Distribution")
plt.tight_layout()
plt.savefig("reports/figures/02_cleaned_feature_distributions.png", dpi=200, bbox_inches="tight")
plt.close()

#================ OBJECTIVE 3: SEPARATING FEATURES AND TARGET ================
# Exclude identifiers, location details, and columns that reveal the churn outcome.
drop_columns = [
    "CustomerID", "target", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude", "Churn Label",
    "Churn Score", "Churn Reason",
]
X = data.drop(columns=drop_columns, errors="ignore")
y = data["target"]

#================ TRAIN TEST SPLIT ================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()
print("Training target distribution:")
print(y_train.value_counts())

#================ NUMERICAL AND CATEGORICAL COLUMNS ================
numerical_columns = X_train.select_dtypes(include="number").columns.tolist()
categorical_columns = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
print("Numerical Columns:")
print(numerical_columns)
print("Categorical Columns:")
print(categorical_columns)

#================ HANDLING MISSING VALUES ================
# Fit the median values on training customers and reuse them for test customers.
numeric_imputer = SimpleImputer(strategy="median")
numeric_imputer.fit(X_train[numerical_columns])
X_train_numeric = pd.DataFrame(
    numeric_imputer.transform(X_train[numerical_columns]),
    columns=numerical_columns,
    index=X_train.index,
)
X_test_numeric = pd.DataFrame(
    numeric_imputer.transform(X_test[numerical_columns]),
    columns=numerical_columns,
    index=X_test.index,
)
print("Missing numerical values after imputation:", X_train_numeric.isnull().sum().sum())

#================ CHECKING OUTLIERS ================
# Review IQR outliers, but retain them because valid charges and tenure values
# should not be removed from the churn dataset solely for being uncommon.
for column in numerical_columns:
    first_quartile = X_train_numeric[column].quantile(0.25)
    third_quartile = X_train_numeric[column].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_limit = first_quartile - 1.5 * interquartile_range
    upper_limit = third_quartile + 1.5 * interquartile_range
    outliers = X_train_numeric[
        (X_train_numeric[column] < lower_limit)
        | (X_train_numeric[column] > upper_limit)
    ]
    print(column, "outliers:", len(outliers))

#================ OUTLIER VISUALIZATION (BOXPLOTS) ================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
boxplot_columns = ["Tenure Months", "Monthly Charges", "Total Charges"]
for position in range(len(boxplot_columns)):
    axis = axes[position]
    column = boxplot_columns[position]
    sns.boxplot(x=X_train_numeric[column], ax=axis)
    axis.set_title(column)
plt.tight_layout()
plt.savefig("reports/figures/03_outlier_boxplots.png", dpi=200, bbox_inches="tight")
plt.close()

#================ LABEL ENCODING GENDER ================
# Gender has two categories, so label encoding keeps this step simple.
gender_fill_value = X_train["Gender"].mode().iloc[0]
X_train_gender = X_train["Gender"].fillna(gender_fill_value).astype(str)
X_test_gender = X_test["Gender"].fillna(gender_fill_value).astype(str)
label_encoder = LabelEncoder()
label_encoder.fit(X_train_gender)
X_train_gender = pd.DataFrame(
    label_encoder.transform(X_train_gender),
    columns=["Gender"],
    index=X_train.index,
)
X_test_gender = pd.DataFrame(
    label_encoder.transform(X_test_gender),
    columns=["Gender"],
    index=X_test.index,
)

#================ ONE HOT ENCODING OTHER CATEGORICAL COLUMNS ================
other_categorical_columns = [
    column for column in categorical_columns if column != "Gender"
]
X_train_categories = X_train[other_categorical_columns].fillna("Unknown").astype(str)
X_test_categories = X_test[other_categorical_columns].fillna("Unknown").astype(str)
X_train_encoded = pd.get_dummies(
    X_train_categories,
    columns=other_categorical_columns,
    dtype="int",
)
X_test_encoded = pd.get_dummies(
    X_test_categories,
    columns=other_categorical_columns,
    dtype="int",
)
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0,
)

dummy_columns = X_train_encoded.columns.tolist()
X_train = pd.concat([X_train_numeric, X_train_gender, X_train_encoded], axis=1)
X_test = pd.concat([X_test_numeric, X_test_gender, X_test_encoded], axis=1)
feature_columns = X_train.columns.tolist()

#================ SCALING NUMERICAL COLUMNS ================
# MinMaxScaler is fitted on training customers and applied to both datasets.
scaler = MinMaxScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[feature_columns] = scaler.fit_transform(X_train[feature_columns])
X_test[feature_columns] = scaler.transform(X_test[feature_columns])

print("Training feature shape after encoding:", X_train.shape)
print("Testing feature shape after encoding:", X_test.shape)

#================ CORRELATION ANALYSIS ================
correlation_data = X_train_numeric.copy()
correlation_data["target"] = y_train
correlation_matrix = correlation_data.corr()
print(correlation_matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Training Customer Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/04_training_correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

#================ OBJECTIVE 4: PREDICT CUSTOMER CHURN ================
#================ TRAINING LOGISTIC REGRESSION ================
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("================ LOGISTIC REGRESSION ================")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_lr, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_lr, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

#================ LOGISTIC REGRESSION CONFUSION MATRIX ================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/figures/05_logistic_regression_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

#================ CHECKING K VALUES FOR KNN ================
X_knn_train, X_knn_validation, y_knn_train, y_knn_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
knn_results = []
for k in range(1, 21):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_knn_train, y_knn_train)
    y_pred_k = model_knn.predict(X_knn_validation)
    k_accuracy = accuracy_score(y_knn_validation, y_pred_k)
    k_precision = precision_score(y_knn_validation, y_pred_k, zero_division=0)
    k_recall = recall_score(y_knn_validation, y_pred_k, zero_division=0)
    k_f1_score = f1_score(y_knn_validation, y_pred_k, zero_division=0)
    knn_results.append({
        "K": k,
        "Accuracy": k_accuracy,
        "Precision": k_precision,
        "Recall": k_recall,
        "F1 Score": k_f1_score,
    })
    print(
        "K =", k,
        "| Accuracy =", round(k_accuracy, 3),
        "| Precision =", round(k_precision, 3),
        "| Recall =", round(k_recall, 3),
        "| F1 Score =", round(k_f1_score, 3),
    )

knn_comparison = pd.DataFrame(knn_results)
knn_comparison = knn_comparison.sort_values("F1 Score", ascending=False)
knn_comparison = knn_comparison.reset_index(drop=True)
best_k = int(knn_comparison.iloc[0]["K"])
print("Selected K by F1 Score:", best_k)

#================ TRAINING KNN ================
model_knn = KNeighborsClassifier(n_neighbors=best_k)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

print("================ K-NEAREST NEIGHBORS ================")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_knn, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_knn, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

#================ TRAINING NAIVE BAYES ================
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

print("================ NAIVE BAYES ================")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_nb, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_nb, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

#================ TRAINING DECISION TREE ================
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

print("================ DECISION TREE ================")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_dt, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_dt, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

#================ TRAINING ADABOOST ================
model_ada = AdaBoostClassifier(random_state=42)
model_ada.fit(X_train, y_train)
y_pred_ada = model_ada.predict(X_test)

print("================ ADABOOST ================")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Precision:", precision_score(y_test, y_pred_ada, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_ada, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_ada, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ada))

#================ SELECTING A MODEL WITH VALIDATION DATA ================
# Keep the test split for final reporting. Choose the model using a separate
# holdout taken only from the training split.
X_selection_train, X_selection_validation, y_selection_train, y_selection_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

selection_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=best_k),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}
validation_f1_scores = {}
for model_name, selection_model in selection_models.items():
    selection_model.fit(X_selection_train, y_selection_train)
    validation_predictions = selection_model.predict(X_selection_validation)
    validation_f1_scores[model_name] = f1_score(
        y_selection_validation, validation_predictions, zero_division=0
    )
best_model_name = max(validation_f1_scores, key=validation_f1_scores.get)
print("Selected model using validation F1:", best_model_name)

#================ OBJECTIVE 5: MODEL COMPARISON ================
model_predictions = {
    "Logistic Regression": y_pred_lr,
    "K-Nearest Neighbors": y_pred_knn,
    "Naive Bayes": y_pred_nb,
    "Decision Tree": y_pred_dt,
    "AdaBoost": y_pred_ada,
}

comparison_rows = [
    {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
    }
    for model_name, predictions in model_predictions.items()
]
comparison = pd.DataFrame(comparison_rows)
comparison = comparison.sort_values("F1 Score", ascending=False)
comparison = comparison.reset_index(drop=True)

print("================ MODEL COMPARISON ================")
print(comparison)
comparison.to_csv("reports/model_comparison.csv", index=False)

#================ MODEL ACCURACY COMPARISON ================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=comparison)
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/figures/06_model_accuracy_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

#================ MODEL F1 SCORE COMPARISON ================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="F1 Score", data=comparison)
plt.title("Model F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/figures/07_model_f1_score_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

#================ CONFUSION MATRIX COMPARISON ================
fig, axes = plt.subplots(1, len(model_predictions), figsize=(5 * len(model_predictions), 4))
if len(model_predictions) == 1:
    axes = [axes]
flat_axes = axes.flatten()
model_names = list(model_predictions.keys())
for position in range(len(model_names)):
    axis = flat_axes[position]
    model_name = model_names[position]
    predictions = model_predictions[model_name]
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_title(model_name)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
for position in range(len(model_names), len(flat_axes)):
    fig.delaxes(flat_axes[position])
    
plt.tight_layout()
plt.savefig("reports/figures/08_model_confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()

#================ SAVING THE BEST MODEL ================
model_dictionary = {
    "Logistic Regression": model_lr,
    "K-Nearest Neighbors": model_knn,
    "Naive Bayes": model_nb,
    "Decision Tree": model_dt,
    "AdaBoost": model_ada,
}
best_model = model_dictionary[best_model_name]
category_options = {}
for column in categorical_columns:
    known_values = X_train_raw[column].fillna("Unknown").astype(str).unique()
    category_options[column] = sorted(known_values.tolist())

joblib.dump(
    {
        "model": best_model,
        "numeric_imputer": numeric_imputer,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "other_categorical_columns": other_categorical_columns,
        "dummy_columns": dummy_columns,
        "feature_columns": feature_columns,
        "selected_input_columns": numerical_columns + categorical_columns,
        "category_options": category_options,
        "threshold": 0.5,
        "model_name": best_model_name,
        "best_k": best_k,
    },
    "models/churn_model.pkl",
)

print("Best model saved:", best_model_name)



