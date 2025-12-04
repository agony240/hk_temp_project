from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_errors(y_true, y_pred_rnn, y_pred_lstm):
    """
    Calculate error metrics for RNN and LSTM predictions compared to actual values.
    Returns a dictionary with MAE, MSE, and RMSE.
    """
    mae_rnn = mean_absolute_error(y_true, y_pred_rnn)
    mae_lstm = mean_absolute_error(y_true, y_pred_lstm)

    mse_rnn = mean_squared_error(y_true, y_pred_rnn)
    mse_lstm = mean_squared_error(y_true, y_pred_lstm)

    rmse_rnn = np.sqrt(mse_rnn)
    rmse_lstm = np.sqrt(mse_lstm)

    return {
        "RNN": {"MAE": mae_rnn, "MSE": mse_rnn, "RMSE": rmse_rnn},
        "LSTM": {"MAE": mae_lstm, "MSE": mse_lstm, "RMSE": rmse_lstm}
    }
