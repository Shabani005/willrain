from flask import Flask, send_file, request, jsonify
import pandas as pd
import importlib
import interface  # your script with full model logic

app = Flask(__name__)


@app.route('/')
def home():
    # Serve index.html directly from this directory
    return send_file("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])
    year = int(data['year'])
    month = int(data['month'])

    # ✅ Rerun your model pipeline from scratch on every click
    global interface
    importlib.reload(interface)

    # Build input for temperature model
    sample = pd.DataFrame([{
        "Longitude": longitude,
        "Latitude": latitude,
        "Year": year,
        "Month": month
    }])

    # Use the trained models from interface
    pred_temp = interface.model.predict(sample)[0]
    rain_prob = interface.rain_model.predict_proba([[pred_temp]])[0, 1]

    # Determine message for the user
    if rain_prob > 0.7:
        message = "It's very likely to rain. 🌧️"
    elif rain_prob > 0.5:
        message = "There's a moderate chance of rain."
    else:
        message = "It's unlikely to rain."

    return jsonify({
        "Predicted_Temperature": round(float(pred_temp), 3),
        "Rain_Probability": round(float(rain_prob), 3),
        "Message": message
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)