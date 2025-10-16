from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load model
model = xgb.XGBRegressor()
model.load_model("XGmodel_enc.json")

@app.route('/')
def home():
    return "Internet Addiction Predictor is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_input = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(X_input)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
