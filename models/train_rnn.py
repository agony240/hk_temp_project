import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "data", "daily_HKO_GMT_ALL.csv")

df = pd.read_csv(file_path, encoding="utf-8-sig", skiprows=2)
df.columns = ["Year", "Month", "Day", "Value", "Completeness"]
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
df = df.dropna(subset=["Date", "Value"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

values = df["Value"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

model = Sequential([
    SimpleRNN(50, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=50, batch_size=32)

model.save(os.path.join(base_dir, "forecast_rnn.h5"))
joblib.dump(scaler, os.path.join(base_dir, "scaler_rnn.pkl"))

print("✅ RNN baseline training complete.")
