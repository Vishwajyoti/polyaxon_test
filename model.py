import argparse
import os
import tempfile

import joblib
from polyaxon import tracking
from polyaxon.tracking.contrib.scikit import log_regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
import pandas as pd

# from cust_feat import *
from google.cloud import bigquery
from feast import FeatureStore, RepoConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    credentials = service_account.Credentials.from_service_account_file(
        "credentials.json", scopes=["https://www.googleapis.com/auth/cloud-platform",],
    )

    tracking.init()

    client = bigquery.Client()
    query = """SELECT * FROM vj-feat-ml.feature_store.cust_demo_det"""
    df_entity = client.query(query).to_dataframe()
    df_entity["event_timestamp"] = pd.Timestamp("2021-07-31", tz="UTC")

    # Initializing Feature Store
    fs = FeatureStore(
        config=RepoConfig(
            registry="gs://cust_feat_new/custfeat.db",
            project="Customer_Feature",
            provider="gcp",
        )
    )
    # Features to be imported
    features = [
        "customer_conn_det:Tenure",
        "customer_conn_det:PhoneService",
        "customer_conn_det:MultipleLines",
        "customer_conn_det:InternetService",
        "customer_conn_det:OnlineSecurity",
        "customer_conn_det:OnlineBackup",
        "customer_conn_det:DeviceProtection",
        "customer_conn_det:TechSupport",
        "customer_conn_det:StreamingTV",
        "customer_conn_det:StreamingMovies",
        "customer_pay_det:Contract",
        "customer_pay_det:PaperlessBilling",
        "customer_pay_det:PaymentMethod",
        "customer_pay_det:MonthlyCharges",
        "customer_pay_det:TotalCharges",
        "customer_pay_det:customer_churn_flag:Churn",
    ]

    # Training DataFrame
    training_df = fs.get_historical_features(
        features=features, entity_df=df_entity
    ).to_df()

    tracking.log_artifact(
        name="data", path="data.csv", kind=V1ArtifactKind.ANY, versioned=False
    )

    X = training_df.drop("Churn", axis=1)
    y = training_df[["Churn"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    tracking.log_data_ref(content=X_train, name="x_train")
    tracking.log_data_ref(content=y_train, name="y_train")
    tracking.log_data_ref(content=X_test, name="x_test")
    tracking.log_data_ref(content=y_test, name="y_test")

    rfr = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth
    )
    rfr.fit(X_train, y_train)

    # Polyaxon
    # This automatically logs metrics relevant to regression
    log_regressor(rfr, X_test, y_test)

    # Logging the model as joblib
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.joblib")
        joblib.dump(rfr, model_path)
        tracking.log_model(
            model_path, name="model", framework="scikit-learn", versioned=False
        )

