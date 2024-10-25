path_python_material = "C:\\Users\\al107\\Downloads\\dsif11_new" # REPLACE WITH YOUR PATH
model_id = "lr1"


# If unsure, print current directory path by executing the following in a new cell:
# !pwd

import numpy as np
import pickle
from fastapi import FastAPI, Query #added query
from pydantic import BaseModel
from fastapi.responses import JSONResponse #new
import plotly.express as px #new
import shap


app = FastAPI()

# Load the pipeline
with open(f"{path_python_material}/models/{model_id}-pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

class Transaction(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float

# Route to get feature importance
@app.get("/feature-importance")
def get_feature_importance():
    importance = loaded_pipeline[1].coef_[0].tolist()
    features = ['transaction_a  mount', 'customer_age', 'customer_balance']
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
    path = f"{path_python_material}/data/2-intermediate/dsif11-X_train_scaled.npy"
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

# Endpoint for feature importance
@app.get("/feature-importance/")
def get_feature_importance():
    # Coefficients for logistic regression
    importance = loaded_pipeline[1].coef_[0]
    feature_names = ["transaction_amount", "customer_age", "customer_balance"]

    # Return feature importance as a dictionary
    feature_importance = dict(zip(feature_names, importance))
    return {"feature_importance": feature_importance}


# Endpoint to create scatter plot based on input features
@app.get("/scatter-plot")
def get_scatter_plot(
    x_feature: str = Query(..., description="Select the feature for the x-axis"),
    y_feature: str = Query(..., description="Select the feature for the y-axis"),
    transaction_amount: float = Query(...),
    customer_age: int = Query(...),
    customer_balance: float = Query(...)
):

    data_dict = {
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "customer_balance": customer_balance
    }

    # check if the selected features exist
    valid_features = ["transaction_amount", "customer_age", "customer_balance"]
    if x_feature not in valid_features or y_feature not in valid_features:
        return JSONResponse(status_code=400, content={"error": "Invalid feature selected"})

    # plot the data
    fig = px.scatter(x=[data_dict[x_feature]], y=[data_dict[y_feature]], 
                     title=f'Scatter plot of {x_feature} vs {y_feature}')
    
    # return the plot
    return JSONResponse(fig.to_json())