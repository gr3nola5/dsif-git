api_url = "http://localhost:8502"

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import plotly.graph_objs as go

st.title("Fraud Detection App")

# Display site header
#image = Image.open("../images/dsif header 2.jpeg")

image_path = "../images/dsif header.jpeg" # changed header from 2 to _
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_column_width=True)  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

data = {
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "customer_balance": customer_balance
    }

if st.button("Show Feature Importance"):
    import matplotlib.pyplot as plt
    response = requests.get(f"{api_url}/feature-importance")
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

if st.button("Predict and show prediction confidence"):
    # Make the API call

    response = requests.post(f"{api_url}/predict/",
                            json=data)
    result = response.json()
    confidence = result['confidence']

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    # Confidence Interval Visualization
    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

if st.button("Predict and show SHAP values"):
    response = requests.post(f"{api_url}/predict/",
                             json=data)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    ######### SHAP #########
    # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("SHAP Values Explanation")

    # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    st.pyplot(fig)
    

# Section for CSV file upload and processing
st.subheader("fraud detection for uploaded file")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Input field for save path
save_path = st.text_input("Enter save path for predictions (e.g., './results/')", "./results/")

if st.button("Upload CSV and Predict"):
    if uploaded_file is not None:
        # Convert the uploaded file to a DataFrame
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_columns = ['transaction_amount', 'customer_age', 'customer_balance', 'transaction_time']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain 'transaction_time', 'transaction_amount', 'customer_age', and 'customer_balance' columns.")
        else:
            # Prepare the file for upload
            files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
            response = requests.post(f"{api_url}/upload-csv/", files=files, data={'save_path': save_path})
            
            # Process the response
            if response.ok:
                result = response.json()
                st.success(result['message'])
                file_path = result['file_path']
                st.markdown(f"[Download Predictions]({file_path})")
            else:
                st.error(response.json().get("error", "An error occurred during file processing."))
    else:
        st.error("Please upload a CSV file.")