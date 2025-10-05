from flask import Flask, send_file, request, jsonify
import pandas as pd
import importlib
import interface  

app = Flask(__name__)


@app.route('/')
def home():
    return send_file("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])
    year = int(data['year'])
    month = int(data['month'])

    global interface
    importlib.reload(interface)

    sample = pd.DataFrame([{
        "Longitude": longitude,
        "Latitude": latitude,
        "Year": year,
        "Month": month
    }])

    pred_temp = interface.model.predict(sample)[0]
    rain_prob = interface.rain_model.predict_proba([[pred_temp]])[0, 1]

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