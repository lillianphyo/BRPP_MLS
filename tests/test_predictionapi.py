import json

def test_predict(client):
    # Define the test data
    test_data = {
        "geo_id": "mandalay",
        "o_rice": 506.26,
        "h_rice": 518.56,
        "l_rice": 480.6,
        "c_rice": 505.03
    }

    # Send POST request to /predict
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')

    # Check if the response status code is 200
    assert response.status_code == 200

    # Parse response data
    data = response.get_json()

    # Check if the response contains the correct keys
    assert 'geo_id' in data
    assert 'input_o_rice' in data
    assert 'predicted_c_rice' in data

    # Validate the geo_id and predicted_c_rice are in the response
    assert data['geo_id'] == test_data['geo_id']
    assert isinstance(data['predicted_c_rice'], float)

def test_predict_invalid_input(client):
    # Send POST request with missing fields
    response = client.post('/predict', data=json.dumps({"geo_id": "mandalay"}), content_type='application/json')

    # Check if the response status code is 400
    assert response.status_code == 400
