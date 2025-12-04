from flask import Flask, render_template
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# 路徑設定
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "daily_HKO_GMT_ALL.csv")

# 讀取資料
df = pd.read_csv(data_path, encoding="utf-8-sig", skiprows=2)
df.columns = ["Year", "Month", "Day", "Value", "Completeness"]
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
df = df.dropna(subset=["Date", "Value"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])
values = df["Value"].values.reshape(-1, 1)

# 建立序列
def create_sequences(data, seq_length=30):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

# 畫圖 function
def plot_forecast(dates, actual, preds, label):
    plt.figure(figsize=(10,5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, preds, label=f"{label} Predicted")
    plt.title(f"{label} Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# 未來預測 function
def forecast_future(model, scaler, values, days=7, seq_length=30):
    last_seq = scaler.transform(values[-seq_length:])
    last_seq = last_seq.reshape(1, seq_length, 1)
    preds = []
    for _ in range(days):
        pred_scaled = model.predict(last_seq)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
    return preds

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["Value"], label="Historical Temperature")
    plt.title("Historical HK Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    return render_template("history.html", plot_url=plot_url)

@app.route("/forecast_rnn")
def forecast_rnn():
    model = load_model(os.path.join(base_dir, "models", "forecast_rnn.h5"))
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler_rnn.pkl"))
    scaled = scaler.transform(values)
    X_test = create_sequences(scaled, seq_length=30)
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    dates = df["Date"].iloc[30:]
    actual = values[30:]
    forecast_table = [{"date": dates.iloc[i].date(), "actual": float(actual[i][0]), "predicted": float(preds[i][0])} for i in range(10)]
    plot_url = plot_forecast(dates, actual, preds, "RNN Baseline")
    return render_template("forecast.html", plot_url=plot_url, model_name="RNN Baseline", forecast_table=forecast_table)

@app.route("/forecast_lstm")
def forecast_lstm():
    model = load_model(os.path.join(base_dir, "models", "forecast_model.h5"))
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    scaled = scaler.transform(values)
    X_test = create_sequences(scaled, seq_length=30)
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    dates = df["Date"].iloc[30:]
    actual = values[30:]
    forecast_table = [{"date": dates.iloc[i].date(), "actual": float(actual[i][0]), "predicted": float(preds[i][0])} for i in range(10)]
    plot_url = plot_forecast(dates, actual, preds, "LSTM Advanced")
    return render_template("forecast.html", plot_url=plot_url, model_name="LSTM Advanced", forecast_table=forecast_table)

@app.route("/forecast_rnn_future")
def forecast_rnn_future():
    model = load_model(os.path.join(base_dir, "models", "forecast_rnn.h5"))
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler_rnn.pkl"))
    preds = forecast_future(model, scaler, values, days=7)
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    forecast_table = [{"date": future_dates[i].date(), "predicted": float(preds[i])} for i in range(7)]
    return render_template("forecast_future.html", model_name="RNN Baseline", forecast_table=forecast_table)

@app.route("/forecast_lstm_future")
def forecast_lstm_future():
    model = load_model(os.path.join(base_dir, "models", "forecast_model.h5"))
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    preds = forecast_future(model, scaler, values, days=7)
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    forecast_table = [{"date": future_dates[i].date(), "predicted": float(preds[i])} for i in range(7)]
    return render_template("forecast_future.html", model_name="LSTM Advanced", forecast_table=forecast_table)

@app.route("/errors")
def show_errors():
    rnn_model = load_model(os.path.join(base_dir, "models", "forecast_rnn.h5"))
    rnn_scaler = joblib.load(os.path.join(base_dir, "models", "scaler_rnn.pkl"))
    scaled_rnn = rnn_scaler.transform(values)
    X_test_rnn = create_sequences(scaled_rnn, seq_length=30)
    preds_rnn = rnn_model.predict(X_test_rnn)
    preds_rnn = rnn_scaler.inverse_transform(preds_rnn)

    lstm_model = load_model(os.path.join(base_dir, "models", "forecast_model.h5"))
    lstm_scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    scaled_lstm = lstm_scaler.transform(values)
    X_test_lstm = create_sequences(scaled_lstm, seq_length=30)
    preds_lstm = lstm_model.predict(X_test_lstm)
    preds_lstm = lstm_scaler.inverse_transform(preds_lstm)

    actual = values[30:]
    mae_rnn = round(mean_absolute_error(actual, preds_rnn), 2)
    rmse_rnn = round(np.sqrt(mean_squared_error(actual, preds_rnn)), 2)
    mae_lstm = round(mean_absolute_error(actual, preds_lstm), 2)
    rmse_lstm = round(np.sqrt(mean_squared_error(actual, preds_lstm)), 2)

    labels = ["RNN", "LSTM"]
    mae_values = [mae_rnn, mae_lstm]
    rmse_values = [rmse_rnn, rmse_lstm]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, mae_values, width, label="MAE", color="skyblue")
    plt.bar(x + width/2, rmse_values, width, label="RMSE", color="salmon")
    plt.xticks(x, labels)
    plt.ylabel("Error Value")
    plt.title("RNN vs LSTM Error Comparison")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_url = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template("errors.html",
                           mae_rnn=mae_rnn, rmse_rnn=rmse_rnn,
                           mae_lstm=mae_lstm, rmse_lstm=rmse_lstm,
                           chart_url=chart_url)

@app.route("/error_range")
def error_range():
    # RNN
    rnn_model = load_model(os.path.join(base_dir, "models", "forecast_rnn.h5"))
    rnn_scaler = joblib.load(os.path.join(base_dir, "models", "scaler_rnn.pkl"))
    scaled_rnn = rnn_scaler.transform(values)
    X_test_rnn = create_sequences(scaled_rnn, seq_length=30)
    preds_rnn = rnn_model.predict(X_test_rnn)
    preds_rnn = rnn_scaler.inverse_transform(preds_rnn)

    # LSTM
    lstm_model = load_model(os.path.join(base_dir, "models", "forecast_model.h5"))
    lstm_scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    scaled_lstm = lstm_scaler.transform(values)
    X_test_lstm = create_sequences(scaled_lstm, seq_length=30)
    preds_lstm = lstm_model.predict(X_test_lstm)
    preds_lstm = lstm_scaler.inverse_transform(preds_lstm)

    actual = values[30:]
    error_rnn = np.abs(actual - preds_rnn)
    error_lstm = np.abs(actual - preds_lstm)

    def count_ranges(errors):
        return {
            "≤2°C": int(np.sum(errors <= 2)),
            "≤5°C": int(np.sum((errors > 2) & (errors <= 5))),
            ">5°C": int(np.sum(errors > 5))
        }

    rnn_stats = count_ranges(error_rnn)
    lstm_stats = count_ranges(error_lstm)

    # Bar chart
    labels = ["≤2°C", "≤5°C", ">5°C"]
    rnn_values = [rnn_stats[l] for l in labels]
    lstm_values = [lstm_stats[l] for l in labels]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, rnn_values, width, label="RNN", color="skyblue")
    plt.bar(x + width/2, lstm_values, width, label="LSTM", color="orange")
    plt.xticks(x, labels)
    plt.ylabel("Number of Days")
    plt.title("Prediction Error Range Comparison")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_url = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template("error_range.html",
                           rnn_stats=rnn_stats,
                           lstm_stats=lstm_stats,
                           chart_url=chart_url)

# ✅ 正確收尾
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
