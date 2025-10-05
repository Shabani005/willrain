import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

data = pd.read_csv("data_comb/combined_data_new.csv")

features = ["Longitude", "Latitude", "Year", "Month"]
target = "AirTemperature"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1
    ))
])

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model performance:")
print(f"  MAE (°K): {mae:.3f}")
print(f"  R² Score: {r2:.3f}")

data["PredictedAirTemp"] = model.predict(data[["Longitude", "Latitude", "Year", "Month"]])

X_rain = data[["PredictedAirTemp"]]
y_rain = data["ItRained"]

X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(
    X_rain, y_rain, test_size=0.2
)


pos_weight = (y_train_rain == 0).sum() / (y_train_rain == 1).sum()

rain_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    scale_pos_weight=pos_weight
)
rain_model.fit(X_train_rain, y_train_rain)

y_pred_rain = rain_model.predict(X_test_rain)
y_prob_rain = rain_model.predict_proba(X_test_rain)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test_rain, y_prob_rain)

best_idx = (2 * precisions * recalls / (precisions + recalls)).argmax()
best_threshold = thresholds[best_idx]

threshold = best_threshold
y_pred_rain_adj = (y_prob_rain >= threshold).astype(int)




recall_1 = recall_score(y_test_rain, y_pred_rain_adj, pos_label=1)
precision_1 = precision_score(y_test_rain, y_pred_rain_adj, pos_label=1)
f1_1 = f1_score(y_test_rain, y_pred_rain_adj, pos_label=1)

print("\n✅ Adjusted threshold performance:")
print(f"  Threshold:       {threshold}")
print(f"  Recall (rain=1): {recall_1:.3f}")
print(f"  Precision (rain=1): {precision_1:.3f}")
print(f"  F1 Score (rain=1): {f1_1:.3f}")


sample = pd.DataFrame([{
    "Longitude": 35.5,
    "Latitude": -1.3,
    "Year": 2025,
    "Month": 12
}])


pred_temp = model.predict(sample)[0]

rain_pred = rain_model.predict([[pred_temp]])[0]
rain_prob = rain_model.predict_proba([[pred_temp]])[0, 1]

print(f"Predicted Temperature: {pred_temp-273.15:.2f}°C")
print(f"Predicted Rain (1=yes, 0=no): {rain_pred}")
print(f"Rain Probability: {rain_prob:.3f}")

if rain_prob > 0.7:
    print("It's very likely to rain.🌧️")
elif rain_prob > 0.5:
    print("There's a moderate chance of rain.")
else:
    print("It's unlikely to rain.")
