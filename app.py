import pandas as pd
import joblib

from fastapi import FastAPI, Request
import uvicorn

ENCODER_PATH = "models/ohe_fraud_encoder.joblib"
MODEL_PATH = "models/xgb_fraud_model.joblib"

app = FastAPI()

@app.get("/")
def health_check():
    return "Healthy!"


@app.post("/fraud-classfier")
async def fraud_prediction(request: Request):
    request_data = await request.json()
    df = pd.DataFrame(request_data)

    # Preprocessing
    categorical_cols = [
        "ProductCD",
        "P_emaildomain",
        "R_emaildomain",
        "card4",
        "M1",
        "M2",
        "M3",
    ]
    X = df[categorical_cols]
    enc = joblib.load(ENCODER_PATH)
    X = pd.DataFrame(
        enc.transform(X).toarray(), columns=enc.get_feature_names_out().reshape(-1)
    )
    X["TransactionAmt"] = df[["TransactionAmt"]].to_numpy()

    # XGBoost Classifier
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X)

    response_map = {0: "Legitimate", 1: "Fraud"}
    return [response_map[prediction] for prediction in pred]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)