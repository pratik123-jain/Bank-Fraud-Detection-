
from pathlib import Path
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, make_response


# -----------------------------------
# Project Configuration
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "Paysim"
    / "PS_20174392719_1491204439457_log.csv"
)

MODEL_PATH = PROJECT_DIR / "models" / "paysim_fraud_model.pkl"
HISTORY_DB = PROJECT_DIR / "prediction_history.db"

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

app = Flask(__name__)


# -----------------------------------
# Initialize Prediction History DB
# -----------------------------------

def init_history_db():
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_class INTEGER NOT NULL,
                probability REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


init_history_db()


# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------
# Dashboard
# -----------------------------------

@app.route("/dashboard")
def dashboard():
    if not DATASET_PATH.exists():
        return (
            "PaySim dataset not found. Check dataset/Paysim folder.",
            404
        )

    df = pd.read_csv(
        DATASET_PATH,
        usecols=["amount", "isFraud"]
    )

    fraud_count = int((df["isFraud"] == 1).sum())
    legitimate_count = int((df["isFraud"] == 0).sum())
    total_transactions = len(df)

    amount_data = df.loc[
        df["amount"] <= df["amount"].quantile(0.99),
        "amount"
    ]

    legitimate_amounts = df.loc[
        df["isFraud"] == 0, "amount"
    ].sample(
        n=min(3000, legitimate_count),
        random_state=42
    )

    fraud_amounts = df.loc[
        df["isFraud"] == 1, "amount"
    ]

    return render_template(
        "dashboard.html",
        total_transactions=f"{total_transactions:,}",
        fraud_count=f"{fraud_count:,}",
        legitimate_count=f"{legitimate_count:,}",
        fraud_rate=round(
            fraud_count / total_transactions * 100, 3
        ),
        legitimate_avg=round(
            df.loc[df["isFraud"] == 0, "amount"].mean(), 2
        ),
        fraud_avg=round(
            df.loc[df["isFraud"] == 1, "amount"].mean(), 2
        ),
        amount_bins=amount_data.tolist(),
        legitimate_amounts=legitimate_amounts.tolist(),
        fraud_amounts=fraud_amounts.tolist()
    )


# -----------------------------------
# Fraud Prediction
# -----------------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    form_values = {}
    result = None
    error = None

    if request.method == "POST":
        form_values = request.form.to_dict()

        try:
            values = {}

            for column in NUMERIC_FEATURES:
                raw_value = request.form.get(column, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter {column}."
                    )

                values[column] = float(raw_value)

                if not pd.notna(values[column]):
                    raise ValueError(
                        f"Please enter a valid value for {column}."
                    )

                if values[column] < 0:
                    raise ValueError(
                        f"{column} cannot be negative."
                    )

            step_value = values["step"]

            if not step_value.is_integer() or step_value < 1:
                raise ValueError(
                    "Transaction Time Step must be a positive integer."
                )

            transaction_type = request.form.get("type", "").strip()
            allowed_types = {
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "CASH_IN",
                "DEBIT",
            }

            if transaction_type not in allowed_types:
                raise ValueError("Please select a valid transaction type.")

            values["step"] = int(step_value)
            values["type"] = transaction_type

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "PaySim model not found. Run "
                    "py src/paysim_train_model.py first."
                )

            model = joblib.load(MODEL_PATH)

            transaction = pd.DataFrame(
                [[values[column] for column in FEATURES]],
                columns=FEATURES
            )

            prediction_value = int(model.predict(transaction)[0])
            probability = round(
                float(model.predict_proba(transaction)[0][1]) * 100,
                2
            )

            result = {
                "prediction": prediction_value,
                "probability": probability,
                "threshold": 50
            }

            # Save prediction to SQLite
            with sqlite3.connect(HISTORY_DB) as conn:
                conn.execute("""
                    INSERT INTO predictions (
                        predicted_class,
                        probability,
                        created_at
                    )
                    VALUES (?, ?, ?)
                """, (
                    prediction_value,
                    probability,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))

        except (ValueError, KeyError) as exc:
            error = str(exc)

        except FileNotFoundError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check the model and entered values."
            )

    return render_template(
        "prediction.html",
        result=result,
        error=error,
        form_values=form_values
    )


# -----------------------------------
# Prediction History
# -----------------------------------

@app.route("/history")
def history():
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.row_factory = sqlite3.Row

        records = conn.execute("""
            SELECT id, predicted_class, probability, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return render_template(
        "history.html",
        records=records
    )


# -----------------------------------
# Download Prediction History CSV
# -----------------------------------

@app.route("/download-history")
def download_history():
    with sqlite3.connect(HISTORY_DB) as conn:
        df = pd.read_sql_query("""
            SELECT
                id AS ID,
                CASE
                    WHEN predicted_class = 1 THEN 'Potential Fraud'
                    ELSE 'Legitimate'
                END AS Prediction,
                probability AS Fraud_Probability_Percent,
                created_at AS Date_Time
            FROM predictions
            ORDER BY id DESC
        """, conn)

    response = make_response(df.to_csv(index=False))
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = (
        "attachment; filename=fraudshield_prediction_history.csv"
    )

    return response


# -----------------------------------
# Run Flask Application
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)
