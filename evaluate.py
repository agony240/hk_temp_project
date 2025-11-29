import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data", "daily_HKO_GMT_ALL.csv")

df = pd.read_csv(file_path, encoding="utf-8-sig", skiprows=2)
df.columns = ["Year", "Month", "Day", "Value", "Completeness"]
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
df = df.dropna(subset=["Date", "Value"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

test = df[(df["Date"] >= "2025-01-01") & (df["Date"] <= "2025-10-30")]

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def evaluate_model(model_path, scaler_path, test_values, label):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    scaled = scaler.transform(test_values.reshape(-1, 1))
    X_test, y_test = create_sequences(scaled, seq_length=30)

    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)

    mae = mean_absolute_error(y_test.reshape(-1,1), preds)
    rmse = sqrt(mean_squared_error(y_test.reshape(-1,1), preds))

    print(f"{label} MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    plt.figure(figsize=(10,5))
    plt.plot(test["Date"].iloc[30:], y_test, label=f"{label} Actual")
    plt.plot(test["Date"].iloc[30:], preds, label=f"{label} Predicted")
    plt.title(f"{label} Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.savefig(f"{label}_forecast.png")
    print(f"✅ Saved plot as {label}_forecast.png")

    # 顯示前 10 日數字
    for i in range(10):
        date = test["Date"].iloc[30+i]
        print(f"{date.date()} | Actual: {y_test[i][0]:.2f} °C | Predicted: {preds[i][0]:.2f} °C")

    plt.show()

evaluate_model(os.path.join(base_dir, "models", "forecast_rnn.h5"),
               os.path.join(base_dir, "models", "scaler_rnn.pkl"),
               test["Value"].values, "RNN")

evaluate_model(os.path.join(base_dir, "models", "forecast_model.h5"),
               os.path.join(base_dir, "models", "scaler.pkl"),
               test["Value"].values, "LSTM")
