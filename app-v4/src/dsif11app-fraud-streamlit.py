api_url = "http://localhost:8503"

import os
import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import FuncFormatter
from io import BytesIO

st.title("Fraud Detection App")

# Display site header
image_path = "../images/dsif header.jpeg"
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

# Section for uploading a CSV file for batch predictions
st.subheader("Upload a CSV file for batch predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Send the file for processing
        with st.spinner('Processing...'):
            response = requests.post(f"{api_url}/upload-csv/", files={"file": uploaded_file.getvalue()})

        # Check if the response was successful
        if response.status_code == 200:

            # Create new column
            df['trx_amnt_to_balance'] = df['transaction_amount'] / df['customer_balance']

            # Step 1: Read the CSV from the binary content (from post request) into a dataframe
            pred = pd.read_csv(BytesIO(response.content))
            pred = pred.round(4)
            # Step 3: Convert dataframe back to CSV format
            processed_csv = pred.to_csv(index=False).encode('utf-8')

            # Preview the uploaded file
            st.write("Preview of uploaded file:")
            st.dataframe(df.head())

            st.success("Predictions completed! You can download the CSV file with predictions.")

            # Get custom file path input
            file_path = st.text_input("Enter the file path to save your CSV", "predictions.csv")

            # Check if file with same name already exists
            if file_path and os.path.exists(file_path):
                overwrite=st.checkbox(f"File '{file_path}' already exists. Overwrite?")

                if overwrite:
                    if st.button("Save new version of file"):
                        try:
                            with open(file_path, 'wb') as f:
                                f.write(processed_csv)
                            st.success(f"File {file_path} has been overwritten successfully.")
                        except Exception as e:
                            st.error(f"Error saving file: {e}")
            else:
            # No overwrite, process and save the file
                if st.button("Save File"):
                    if file_path:
                        try:
                            with open(file_path, "wb") as f:
                                f.write(processed_csv)
                            st.success(f"File {file_path} has been saved successfully.")
                        except Exception as e:
                            st.error(f"Error saving file: {e}")
                else:
                    # If no file path was entered, show an error message
                    st.error("Please provide a valid file path.")
        else:
            error_message = response.json().get("error", "Unknown error occurred")
            st.error(f"Error: {error_message}")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {str(e)}")


# Only enable scatter plot section if the file is uploaded and the dataframe (df) is available
if uploaded_file is not None:
    # Function to add tick values to comma separators
    def format_ticks(x, _):
        return f'{int(x):,}'  # commas

    # Scatter plot feature
    st.subheader("Interactive Scatter Plot")

    # Allow users to select columns for x and y axes
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'transaction_amount' in numeric_columns and 'customer_balance' in numeric_columns:
        x_axis = st.selectbox("Choose x-axis", options=numeric_columns, index=numeric_columns.index('transaction_amount'))
        y_axis = st.selectbox("Choose y-axis", options=numeric_columns, index=numeric_columns.index('customer_balance'))

        # Apply a matplotlib theme
        plt.style.use('bmh')

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[x_axis], df[y_axis], alpha=0.7, edgecolors='w', linewidth=0.5)

        # title and labels
        ax.set_title(f'{y_axis.replace("_", " ")} by {x_axis.replace("_", " ")}', fontsize=12, color='black')
        ax.set_xlabel(x_axis.replace("_", " "), fontsize=10, color='black')
        ax.set_ylabel(y_axis.replace("_", " "), fontsize=10, color='black')
        ax.tick_params(axis='both', which='major', labelsize=8, color='black')
        ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))  # Format x-axis ticks
        ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))  # Format y-axis ticks
        ax.grid(False)  # no grid lines
        st.pyplot(fig)
