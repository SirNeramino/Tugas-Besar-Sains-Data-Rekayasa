import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor 

from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

class RNNWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        X_arr = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X_arr, verbose=0).flatten()

class SARIMAWrapper:
    def __init__(self, model_fit):
        self.model_fit = model_fit
    def predict(self, X):
        return self.model_fit.forecast(steps=len(X), exog=X).values

def load_real_data(dataset_name):
    try:
        if "Performa Turbin" in dataset_name:
            df = pd.read_csv('turbine_speed_direction.csv')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df.dropna()
        
        elif "Keandalan Gearbox" in dataset_name:
            df = pd.read_csv('turbine_gearbox.csv')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df.dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_regression_fungsi1(metode, parameters, df):
    features = ['WS_10m', 'WS_100m', 'WD_100m', 'Temp_2m', 'RelHum_2m']
    X = df[features]
    y = df['Power']
    
    if 'Time' in df.columns:
        time_series = df['Time']
    elif 'timestamp' in df.columns:
        time_series = df['timestamp']
    else:
        time_series = pd.Series(df.index)
    
    if metode == "SARIMA (Time-Series)":
        X = X.tail(2000)
        y = y.tail(2000)
        time_series = time_series.tail(2000)
    elif metode == "Recurrent Neural Network (RNN)":
        X = X.tail(10000)
        y = y.tail(10000)
        time_series = time_series.tail(10000)
    
    train_ratio = parameters.get('train_size', 80) / 100.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)
    time_train, time_test = train_test_split(time_series, train_size=train_ratio, shuffle=False)
    
    feat_importance = None
    
    if metode == "SARIMA (Time-Series)":
        model_sm = SARIMAX(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model_sm.fit(disp=False)
        y_pred = model_fit.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test).values
        model = SARIMAWrapper(model_fit)
        
    elif metode == "Recurrent Neural Network (RNN)":
        X_train_rnn = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_rnn = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        rnn_model = Sequential()
        rnn_model.add(SimpleRNN(32, input_shape=(1, X_train.shape[1]), activation='relu'))
        rnn_model.add(Dense(16, activation='relu'))
        rnn_model.add(Dense(1))
        
        rnn_model.compile(optimizer='adam', loss='mse')
        epochs = parameters.get('epochs', 50)
        rnn_model.fit(X_train_rnn, y_train, epochs=epochs, batch_size=128, verbose=0)
        
        y_pred = rnn_model.predict(X_test_rnn, verbose=0).flatten()
        model = RNNWrapper(rnn_model)
        
    elif metode == "XGBoost Regressor":
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': model.feature_importances_})
        
    elif metode == "Extra Trees Regressor":
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': model.feature_importances_})
        
    elif metode == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': model.feature_importances_})
        
    elif metode == "Multi-Layer Perceptron (MLP)":
        model = MLPRegressor(max_iter=parameters.get('epochs', 200), random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else: 
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    y_pred = np.maximum(y_pred, 0)
        
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test + 1e-10, y_pred)
    
    residuals = y_test.values - y_pred
    
    chart_df = pd.DataFrame({
        'Waktu': pd.to_datetime(time_test.values, errors='coerce'), 
        'Daya Aktual': y_test.values, 
        'Prediksi Daya': y_pred, 
        'Galat (Residual)': residuals
    })
    
    if feat_importance is not None:
        feat_importance = feat_importance.sort_values(by='Tingkat Kepentingan', ascending=True)

    return {
        "metrik1_nama": "R² Score", "metrik1_nilai": f"{r2:.4f}",
        "metrik2_nama": "RMSE (MW)", "metrik2_nilai": f"{rmse:.4f}",
        "metrik3_nama": "MAE (MW)", "metrik3_nilai": f"{mae:.4f}",
        "metrik4_nama": "MAPE (%)", "metrik4_nilai": f"{mape*100:.2f}%",
        "status": "Sukses",
        "chart_data": chart_df,
        "feature_importance": feat_importance,
        "model": model,           
        "fitur": features         
    }

def run_anomaly_fungsi2(metode, parameters, df):
    # FUNGSI 2 DIKOSONGKAN SEMENTARA SESUAI PERMINTAAN
    return {
        "metrik1_nama": "-", "metrik1_nilai": "-",
        "metrik2_nama": "-", "metrik2_nilai": "-",
        "metrik3_nama": "-", "metrik3_nilai": "-",
        "metrik4_nama": "-", "metrik4_nilai": "-",
        "status": "Dikosongkan",
        "chart_data": pd.DataFrame(), 
        "model": None,           
        "fitur": []         
    }