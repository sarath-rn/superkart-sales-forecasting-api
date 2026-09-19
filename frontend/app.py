import streamlit as st
import requests
import pandas as pd
import json
import os

# Configure page layout
st.set_page_config(
    page_title="SuperKart Retail Sales Predictor",
    layout="wide"
)

st.title("SuperKart Retail Sales Forecasting")
st.markdown("Predict total product sales across store locations using the trained XGBoost model.")

# Define backend API URL (Uses Docker network alias or local fallback)
BACKEND_URL = os.getenv("BACKEND_URL", 
# "http://backend:5000/predict"
"http://superkart-backend-container:5000/predict"
)

# Navigation sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Inference Mode", ["Single Prediction", "Batch Prediction"])

# 1. SINGLE PREDICTION MODE
if app_mode == "Single Prediction":
    st.subheader("Single Product Sales Inference")
    st.write("Enter product and store characteristics to estimate Product_Store_Sales_Total.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Product Details")
        product_id = st.text_input("Product ID", value="FD102")
        product_weight = st.number_input("Product Weight (kg)", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
        product_allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        product_mrp = st.number_input("Product MRP ($)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_type = st.selectbox(
            "Product Type",
            [
                "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
                "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks",
                "Meat", "Breads", "Hard Drinks", "Starchy Foods", "Breakfast", "Seafood", "Others"
            ]
        )

    with col2:
        st.markdown("### Store Details")
        store_id = st.text_input("Store ID", value="OUT027")
        store_establishment_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2026, value=2005, step=1)
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox(
            "Store Type",
            ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
        )

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Product_Id": product_id,
            "Product_Weight": product_weight,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Type": product_type,
            "Store_Id": store_id,
            "Store_Establishment_Year": store_establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type
        }

        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                predicted_sales = result["predictions"][0]
                st.success(f"### Predicted Total Sales: **${predicted_sales:,.2f}**")
            else:
                st.error(f"Backend returned error code {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend API at `{BACKEND_URL}`. Details: {str(e)}")

# 2. BATCH PREDICTION MODE
elif app_mode == "Batch Prediction":
    st.subheader("Batch Sales Inference via CSV Upload")
    st.write("Upload a CSV file containing multiple product/store records to run batch predictions.")

    uploaded_file = st.file_uploader("Upload Input CSV File", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Input Data Preview")
        st.dataframe(input_df.head(10))

        if st.button("Run Batch Inference", type="primary"):
            payload = input_df.to_dict(orient="records")

            try:
                response = requests.post(BACKEND_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    predictions = result["predictions"]

                    output_df = input_df.copy()
                    output_df["Predicted_Product_Store_Sales_Total"] = predictions

                    st.write("### Prediction Results")
                    st.dataframe(output_df.head(10))

                    # Download link for predictions
                    csv_data = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="superkart_sales_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Backend returned error code {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend API at `{BACKEND_URL}`. Details: {str(e)}")
