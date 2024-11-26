# app/app.py
from flask import Flask, request, jsonify
import os
import sys
from . import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.predict import predict_rice_price

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract geo_id and c_rice from the request
    geo_id = data.get('geo_id')
    o_rice = data.get('o_rice')
    h_rice = data.get('h_rice')
    l_rice = data.get('l_rice')
    c_rice = data.get('c_rice')

    # Validate input
    if geo_id is None or c_rice is None:
        return jsonify({'error': 'Invalid input, please provide geo_id and c_rice'}), 400

    try:
        # Perform prediction
        predicted_price = predict_rice_price(geo_id, o_rice, h_rice, l_rice, c_rice)

        # Return the prediction
        return jsonify({
            'geo_id': geo_id,
            'input_o_rice': o_rice,
            'input_h_rice': h_rice,
            'input_l_rice': l_rice,
            'input_c_rice': c_rice,
            'predicted_c_rice': predicted_price
        }), 200
        
        # # Return the prediction
        # return jsonify({
        #     'geo_id': geo_id,
        #     'input_c_rice': c_rice,
        #     'predicted_c_rice': predicted_price
        # }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500