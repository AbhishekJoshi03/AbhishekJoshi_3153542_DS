# Telco Customer Attrition Predictor

## 1. Project Overview

This project develops a machine learning model that predicts whether a
telecom customer is likely to churn.

Customer demographic, service, contract, tenure and billing information is
used to estimate the probability of attrition. A Decision Tree Classifier
was trained inside a scikit-learn pipeline and exposed through a Flask REST
API. A small browser console is served by the same application for manual
scoring.

---

## 2. Project Layout

### 2.1 Folder structure

```text
telco_attrition_predictor/
│
├── artifacts/
│   └── attrition_pipeline.pkl        trained scikit-learn pipeline
│
├── data/
│   ├── telco_customers.csv           raw dataset (7,043 rows, 21 columns)
│   └── telco_data_dictionary.csv     column descriptions and project field names
│
├── interface/
│   └── console.html                  browser console served at "/"
│
├── notebooks/
│   └── attrition_study.ipynb         EDA, feature engineering, training, export
│
├── server.py                         Flask application
├── sample_payload.json               example request body for /predict
├── requirements.txt                  runtime dependencies
├── requirements-notebook.txt         runtime + notebook dependencies
└── README.md
```

### 2.2 Training flow

```text
data/telco_customers.csv
        │
        ▼
 rename columns ──► convert total_billed ──► add mean_monthly_fee
                                             and active_service_count
        │
        ▼
 70/30 stratified split (random_state = 42)
        │
        ▼
 column_processor ──► tree_stage (Decision Tree)
   numeric  : median imputation
   category : most-frequent imputation + one-hot encoding
        │
        ▼
 compare depth 4 and depth 8 ──► keep depth 8
        │
        ▼
 artifacts/attrition_pipeline.pkl
```

### 2.3 Request flow

```text
  Browser console          curl / Postman / other clients
        │                              │
        └──────────────┬───────────────┘
                       │  POST /predict   (JSON, 21 fields)
                       ▼
        ┌──────────────────────────────────┐
        │ server.py  ·  assess_customer()  │
        │  1. read JSON body               │
        │  2. check every INPUT_FIELDS key │
        │  3. build a one-row DataFrame    │
        └────────────────┬─────────────────┘
                         ▼
        ┌──────────────────────────────────┐
        │ attrition_pipeline.pkl           │
        │  column_processor ► tree_stage   │
        └────────────────┬─────────────────┘
                         ▼
   { "attrition_flag": "Yes" | "No", "attrition_probability": 0.5775 }
```

### 2.4 Console layout

```text
┌────────────────────────────────────────────────────────────────────┐
│  Attrition Risk Console                             ● API running  │
├──────────────────────────────────────────────┬─────────────────────┤
│  CUSTOMER PROFILE                            │  ASSESSMENT         │
│  [Gender] [Senior] [Partner] [Dependents]    │                     │
│                                              │      ╭───────╮      │
│  SERVICES                                    │     ╱  57.8%  ╲     │
│  [Phone] [Multiple lines] [Internet plan]    │    churn probability│
│  [Security] [Backup] [Protection] [Support]  │                     │
│  [Streaming TV] [Streaming movies]           │  [ Likely to churn ]│
│                                              │                     │
│  CONTRACT AND BILLING                        │  guidance note      │
│  [Contract] [Paperless] [Payment method]     │                     │
│  [Tenure] [Monthly fee] [Total billed]       │  ▸ Request/Response │
│  ┌ average monthly fee ┐ ┌ active services ┐ │    JSON + copy      │
│                                              │                     │
│           [Reset] [Load sample] [Assess]     │                     │
└──────────────────────────────────────────────┴─────────────────────┘
```

On screens narrower than 900 px the assessment panel moves below the form.

---

## 3. Objective

The main objectives of this project are:

- Understand and prepare the customer churn dataset.
- Perform exploratory data analysis.
- Create meaningful engineered features.
- Train Decision Tree classification models.
- Compare different Decision Tree configurations.
- Evaluate the final model using classification metrics.
- Identify important predictors of customer churn.
- Save the trained model.
- Expose the model through a Flask REST API.

---

## 4. Machine Learning Model

Two Decision Tree configurations were evaluated.

### Model 1

- Algorithm: Decision Tree Classifier
- Maximum depth: 4
- Random state: 42

### Model 2

- Algorithm: Decision Tree Classifier
- Maximum depth: 8
- Minimum samples per leaf: 10
- Random state: 42

Model 2 was selected as the final model because it achieved better
Recall and F1 Score, which are important for identifying customers
who are likely to churn.

---

## 5. Feature Engineering

Two additional features were created.

### mean_monthly_fee

Calculated as:

total_billed / tenure_months

It provides an estimate of the customer's average monthly spending.
Customers with zero tenure are treated as missing and imputed by the
pipeline.

### active_service_count

The number of subscribed services used by the customer.

It is calculated by counting the service columns that have a value of
"Yes": `phone_line`, `multi_line`, `security_addon`, `backup_addon`,
`protection_addon`, `support_addon`, `tv_streaming` and `movie_streaming`.

---

## 6. Field Reference

Column names were renamed consistently across the notebook, the trained
pipeline, the API and the console. Category values such as `Yes`, `No`,
`Fiber optic` or `Month-to-month` are unchanged.

| Original column | Project field | Notes |
|---|---|---|
| customerID | client_id | identifier, dropped before training |
| gender | client_gender | |
| SeniorCitizen | senior_flag | 0 or 1 |
| Partner | has_partner | |
| Dependents | has_dependents | |
| tenure | tenure_months | |
| PhoneService | phone_line | |
| MultipleLines | multi_line | |
| InternetService | internet_plan | |
| OnlineSecurity | security_addon | |
| OnlineBackup | backup_addon | |
| DeviceProtection | protection_addon | |
| TechSupport | support_addon | |
| StreamingTV | tv_streaming | |
| StreamingMovies | movie_streaming | |
| Contract | contract_term | |
| PaperlessBilling | ebilling | |
| PaymentMethod | payment_mode | |
| MonthlyCharges | monthly_fee | |
| TotalCharges | total_billed | |
| Churn | attrition | target, not sent to the API |
| AvgMonthlySpend | mean_monthly_fee | engineered |
| NumServices | active_service_count | engineered |

---

## 7. Final Model Performance

The final Decision Tree achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 77.99% |
| Precision - Churn | 60.04% |
| Recall - Churn | 51.16% |
| F1 Score - Churn | 55.25% |

The confusion matrix was:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 1361 | 191 |
| Actual Churn | 274 | 287 |

The model correctly identified 287 customers who actually churned.

Recall was considered particularly important because a false negative
represents a customer who churns but was not identified by the model.

---

## 8. Important Features

The most important features identified by the final Decision Tree were:

1. contract_term - Month-to-month (0.395)
2. tenure_months (0.159)
3. internet_plan - Fiber optic (0.118)
4. mean_monthly_fee (0.060)
5. monthly_fee (0.055)
6. total_billed (0.044)
7. payment_mode - Electronic check (0.025)
8. support_addon - No (0.021)

Contract type, particularly month-to-month contracts, was the most
important predictor in the trained model.

Feature importance represents the contribution of a feature to the
model's predictions and does not imply that the feature directly causes
customer churn.

---

## 9. Getting Started

### 9.1 Run the API and console

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000` in a browser for the console. The same address
returns a JSON status message to non-browser clients.

### 9.2 Rebuild the model

```bash
pip install -r requirements-notebook.txt
cd notebooks
jupyter notebook attrition_study.ipynb
```

Run all cells. The final cells write `artifacts/attrition_pipeline.pkl`.
The notebook must be started from the `notebooks/` folder because the data
and artifact paths are relative.

`requirements.txt` pins `scikit-learn==1.7.2`, the version the bundled
pipeline was saved with. Pickled pipelines should be loaded with the same
scikit-learn version that produced them.

---

## 10. API Reference

### GET /

Returns the console when the client asks for HTML, otherwise:

```json
{
  "service": "Telco Attrition Predictor",
  "status": "running"
}
```

### POST /predict

Send a JSON body containing all 21 fields listed below.

```json
{
    "client_gender": "Female",
    "senior_flag": 0,
    "has_partner": "Yes",
    "has_dependents": "No",
    "tenure_months": 5,
    "phone_line": "Yes",
    "multi_line": "No",
    "internet_plan": "Fiber optic",
    "security_addon": "No",
    "backup_addon": "No",
    "protection_addon": "No",
    "support_addon": "No",
    "tv_streaming": "Yes",
    "movie_streaming": "Yes",
    "contract_term": "Month-to-month",
    "ebilling": "Yes",
    "payment_mode": "Electronic check",
    "monthly_fee": 80.5,
    "total_billed": 402.5,
    "mean_monthly_fee": 80.5,
    "active_service_count": 3
}
```

Successful response:

```json
{
  "attrition_flag": "Yes",
  "attrition_probability": 0.5775
}
```

Error responses use HTTP 400:

| Cause | Body |
|---|---|
| Empty, non-JSON or non-object body | `{"error": "Request body must contain JSON data"}` |
| One or more fields absent | `{"error": "Missing required fields", "missing_fields": ["tenure_months"]}` |
| Value the pipeline cannot process | `{"error": "<message from the pipeline>"}` |

### Example calls

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_payload.json
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post `
    -ContentType "application/json" -InFile sample_payload.json
```

---

## 11. Console Notes

- The two calculated fields, `mean_monthly_fee` and `active_service_count`,
  are filled in automatically using the same rules as the notebook.
- Choosing "No" for phone service sets multiple lines to "No phone service".
  Choosing "No" for internet plan sets the six internet add-ons to
  "No internet service". These options are locked to match the dataset.
- Tenure must be at least 1 month because the average monthly fee divides by
  tenure.
- The request and response panel shows the exact JSON sent to `/predict`,
  with a button to copy it.
- Light and dark appearance follow the operating system setting.
