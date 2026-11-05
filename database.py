"""Functions for saving and managing prediction history in MySQL."""

import json

import mysql.connector

from auth import get_setting


def connect_to_database():
    """Open a connection to the MySQL server using private app settings."""
    database_settings = {
        "host": get_setting("DB_HOST", "localhost"),
        "port": int(get_setting("DB_PORT", "3306")),
        "user": get_setting("DB_USER"),
        "password": get_setting("DB_PASSWORD"),
        "database": get_setting("DB_NAME", "customer_churn_db"),
    }

    if not database_settings["user"]:
        raise RuntimeError(
            "Database username is missing. Add DB_USER to .streamlit/secrets.toml."
        )

    return mysql.connector.connect(**database_settings)


def create_history_table():
    """Create the prediction history table the first time it is needed."""
    connection = connect_to_database()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(40) NOT NULL,
            name VARCHAR(255),
            email VARCHAR(255),
            age INT,
            created_at VARCHAR(40) NOT NULL,
            prediction TINYINT NOT NULL,
            predicted_label VARCHAR(40) NOT NULL,
            probability_churn DECIMAL(7, 4) NOT NULL,
            threshold_value DECIMAL(7, 4) NOT NULL,
            features JSON NOT NULL
        )
        """
    )

    connection.commit()
    cursor.close()
    connection.close()


def save_prediction(record):
    """Save one prediction and return it with the database record ID."""
    create_history_table()
    connection = connect_to_database()
    cursor = connection.cursor()

    insert_query = """
        INSERT INTO prediction_history (
            customer_id, name, email, age, created_at, prediction,
            predicted_label, probability_churn, threshold_value, features
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        record.get("customer_id", ""),
        record.get("name", ""),
        record.get("email", ""),
        record.get("age"),
        record.get("created_at", ""),
        record.get("prediction", 0),
        record.get("predicted_label", ""),
        record.get("probability_churn", 0),
        record.get("threshold", 0.5),
        json.dumps(record.get("features", {}), ensure_ascii=False),
    )

    cursor.execute(insert_query, values)
    record["id"] = cursor.lastrowid
    connection.commit()
    cursor.close()
    connection.close()
    return record


def get_predictions():
    """Return the newest saved predictions first."""
    create_history_table()
    connection = connect_to_database()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT id, customer_id, name, email, age, created_at, prediction,
               predicted_label, probability_churn, threshold_value, features
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 500
        """
    )
    records = cursor.fetchall()

    for record in records:
        if isinstance(record["features"], str):
            record["features"] = json.loads(record["features"])
        record["threshold"] = float(record.pop("threshold_value"))
        record["probability_churn"] = float(record["probability_churn"])

    cursor.close()
    connection.close()
    return records


def delete_prediction(record_id):
    """Delete one saved prediction by its database ID."""
    create_history_table()
    connection = connect_to_database()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM prediction_history WHERE id = %s", (record_id,))
    connection.commit()
    cursor.close()
    connection.close()


def delete_predictions(record_ids=None):
    """Delete selected records, or clear the table when no IDs are given."""
    create_history_table()
    connection = connect_to_database()
    cursor = connection.cursor()

    if record_ids:
        placeholders = ", ".join(["%s"] * len(record_ids))
        query = f"DELETE FROM prediction_history WHERE id IN ({placeholders})"
        cursor.execute(query, tuple(record_ids))
    else:
        cursor.execute("DELETE FROM prediction_history")

    connection.commit()
    cursor.close()
    connection.close()
