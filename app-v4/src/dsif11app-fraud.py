

path_python_material = "C:\\Users\\al107\\Downloads\\dsif11_new" # REPLACE WITH YOUR PATH
model_id = "lr1"


# If unsure, print current directory path by executing the following in a new cell:
# !pwd

import numpy as np
import pandas as pd  # Import pandas for handling CSV files
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import shap
import io


app = FastAPI()

# Load the pipeline
with open(f"{path_python_material}\\app-v4\\models\\{model_id}-pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

class Transaction(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float

# Route to get feature importance
@app.get("/feature-importance")
def get_feature_importance():
    importance = loaded_pipeline[1].coef_[0].tolist()
    features = ['transaction_amount', 'customer_age', 'customer_balance']
    feature_importance = dict(zip(features, importance))
    return {"feature_importance": feature_importance}

# Route to predict new observations
@app.post("/predict/")
def predict_fraud(transaction: Transaction):

    data_point = np.array([[
        transaction.transaction_amount,
        transaction.customer_age,
        transaction.customer_balance
    ]])

    # Make predictions
    prediction = loaded_pipeline.predict(data_point)

    # Get probabilities for each class
    probabilities = loaded_pipeline.predict_proba(data_point)
    confidence = probabilities[0].tolist()

    # Shap values
    path = f"{path_python_material}\\app-v4\\data\\2-intermediate\\dsif11-X_train_scaled.npy"
    print(path)
    X_train_scaled = np.load(path)
    explainer = shap.LinearExplainer(loaded_pipeline[1], X_train_scaled)
    shap_values = explainer.shap_values(data_point)
    print("SHAP", shap_values.tolist())

    return {
        "fraud_prediction": int(prediction[0]),
        "confidence": confidence,
        "shap_values": shap_values.tolist(),
        "features": ['transaction_amount', 'customer_age', 'customer_balance']
        }

# New route to upload CSV and get predictions for multiple transactions
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Read the CSV file into a dataframe
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty"})

        df = pd.read_csv(io.BytesIO(contents))

        # Validate required columns
        required_columns = ['transaction_amount', 'customer_age', 'customer_balance', 'transaction_time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return JSONResponse(status_code=400, content={"error": f"Uploaded file is missing required columns: {', '.join(missing_columns)}"})

        # Create new column
        df['trx_amnt_to_balance'] = df['transaction_amount'] / df['customer_balance']

        # Prepare data for prediction
        data_for_prediction = df[['transaction_amount', 'customer_age', 'customer_balance']].values
        # Predict probabilities
        predictions = loaded_pipeline.predict_proba(data_for_prediction)
        # Add the fraud probability as a new column
        df['fraud_probability'] = predictions[:, 1]  # Assuming column index 1 represents fraud probability

        # Convert the dataframe to CSV and send it back
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
