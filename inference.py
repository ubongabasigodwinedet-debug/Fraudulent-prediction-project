import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from schema import Transaction

#load the trained model and preprocessor

rf_model = joblib.load('notebook\rf_model.joblib')
preprocessor = joblib.load('notebook\preprocessor.joblib')

#instantiate the FastAPI app
app = FastAPI(title="Fraudulent Prediction API")

#define the input data model

class Transaction(BaseModel):
    home_country: str = Field(example="us")
    source_currency: str = Field(example="usd")
    dest_currency: str = Field(example="cad")
    channel: str = Field(example="web")
    amount_src: float = Field(example=278.19)
    amount_usd: float = Field(example=278.19)
    fee: float = Field(example=4.25)
    exchange_rate_src_to_dest: float = Field(example=1.35)
    new_device: int = Field(example=0)
    ip_country: str = Field(example="us")
    location_mismatch: int = Field(example=0)
    ip_risk_score: float = Field(example=0.123)
    kyc_tier: str = Field(example="standard")
    account_age_days: int = Field(example=263)
    device_trust_score: float = Field(example=0.522)
    chargeback_history_count: int = Field(example=0)
    risk_score_internal: float = Field(example=0.223)
    txn_velocity_1h: int = Field(example=0)
    txn_velocity_24h: int = Field(example=1)
    corridor_risk: float = Field(example=0.0)
    hour: int = Field(example=18)
    day_of_week: int = Field(example=0)
    is_weekend: int = Field(example=0)
    age_bucket: str = Field(example="180-365d")
    amount_bucket: str = Field(example="$100-500")
    ip_risk_bucket: str = Field(example="Low<0.3")
    device_trust_bucket: str = Field(example="0.5-0.7")

    # Define the prediction endpoint

    @app.post("/predict")
    def predict_fraud(transaction: Transaction):
        # Convert the input data to a DataFrame
        df = pd.DataFrame([transaction.model_dump()])

        # Preprocess the input data
        df_processed = preprocessor.transform(df)
    
        # Make the prediction
        prediction = int(rf_model.predict(df_processed)[0])

        # Return the prediction result
        return {
            "is_fraud": prediction,
            "label":"fraud" if prediction == 1 else "legit"
        }
    




