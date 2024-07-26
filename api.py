from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('saved_model/model.h')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        features_array = np.array(features).reshape(1, -1)  # Reshape if necessary
        prediction = model.predict(features_array)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
