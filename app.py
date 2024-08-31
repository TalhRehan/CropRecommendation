import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Get input from form
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare input features
        feature_list = [N, P, K, temp, humidity, pH, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Predict using the model
        predict = model.predict(single_pred)

        # Mapping model output to crop names
        crop_dict = {
            1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
            6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
            11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
            16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
            20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
        }

        # Ensure the prediction is within the crop_dict range
        if predict[0] in crop_dict:
            crop = crop_dict[predict[0]]
            result = "{} is the best crop to be cultivated.".format(crop)
        else:
            result = "Sorry, we are not able to recommend a proper crop for this environment."

        return render_template('index.html', result=result)

    except Exception as e:
        # Handle any error and display a user-friendly message
        error_message = "An error occurred: {}".format(e)
        return render_template('index.html', result=error_message)


# Main function
if __name__ == "__main__":
    app.run(debug=True, port=5004)
