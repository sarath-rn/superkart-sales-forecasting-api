from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the serialized model pipeline
MODEL_PATH = 'superkart_model.joblib'

try:
    model = joblib.load(MODEL_PATH)
    print("Serialized XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "SuperKart Retail Sales Prediction Backend API is running."
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on server startup."}), 500

    try:
        data = request.get_json()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid JSON format. Expected object or array."}), 400

        # Feature preprocessing matching notebook transformations
        if 'Product_Sugar_Content' in df.columns:
            df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg': 'Regular'})

        if 'Store_Age_Years' not in df.columns and 'Store_Establishment_Year' in df.columns:
            df['Store_Age_Years'] = 2026 - df['Store_Establishment_Year']

        if 'Product_Id_char' not in df.columns and 'Product_Id' in df.columns:
            df['Product_Id_char'] = df['Product_Id'].astype(str).str[:2]

        if 'Product_Type_Category' not in df.columns and 'Product_Type' in df.columns:
            perishable_items = ['Fruits and Vegetables', 'Meat', 'Dairy', 'Breads', 'Seafood', 'Breakfast']
            df['Product_Type_Category'] = df['Product_Type'].apply(
                lambda x: 'Perishables' if x in perishable_items else 'Non Perishables'
            )

        cols_to_drop = ['Product_Id', 'Store_Id', 'Product_Type', 'Store_Establishment_Year']
        df_model_input = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Generate prediction
        predictions = model.predict(df_model_input)
        predictions_list = [float(np.round(pred, 2)) for pred in predictions]

        return jsonify({
            "status": "success",
            "predictions": predictions_list
        }), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
