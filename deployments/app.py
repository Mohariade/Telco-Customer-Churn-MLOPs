from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_logreg.pkl")


try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ DEPLOYMENT: Loaded model from {MODEL_PATH}")
    else:
        print(f"❌ DEPLOYMENT ERROR: Model not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"❌ DEPLOYMENT ERROR: {e}")
    model = None
    
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        
        
        trained_features = model.feature_names_in_
        
        
        for col in trained_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        
        input_df = input_df[trained_features]
        

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[:, 1][0]
        
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': round(float(probability), 4),
            'status': 'Success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'Failed'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'model_path': MODEL_PATH})

if __name__ == '__main__':
    print("🚀 Starting Flask Deployment Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
