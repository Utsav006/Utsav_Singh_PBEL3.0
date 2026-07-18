from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model and scaler
model_path = os.path.join('..', 'models', 'predictive_maintenance_rf.joblib')
scaler_path = os.path.join('..', 'models', 'scaler.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    # Renders the HTML UI
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture inputs from the HTML form
    machine_type = request.form['type']
    air_temp = float(request.form['air_temp'])
    process_temp = float(request.form['process_temp'])
    rotational_speed = float(request.form['rotational_speed'])
    torque = float(request.form['torque'])
    tool_wear = float(request.form['tool_wear'])

    # 2. Map the Machine Type (L, M, H) back to the LabelEncoder integers (H=0, L=1, M=2)
    type_mapping = {'H': 0, 'L': 1, 'M': 2}
    type_encoded = type_mapping[machine_type]

    # 3. Format the data as a 2D array exactly how the model expects it
    input_data = np.array([[type_encoded, air_temp, process_temp, rotational_speed, torque, tool_wear]])

    # 4. Scale the input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # 5. Make the prediction
    prediction = model.predict(input_scaled)
    
    # 6. Interpret the result
    result = "WARNING: Imminent Machine Failure Detected!" if prediction[0] == 1 else "Normal Operation"
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)