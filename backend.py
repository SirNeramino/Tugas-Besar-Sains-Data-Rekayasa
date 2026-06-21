import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# ==========================================
# CLASS WRAPPERS UNTUK SIMULATOR FRONTEND
# ==========================================

class MLWrapper:
    """Wrapper untuk algoritma Machine Learning standar (XGBoost, RF, SVR, MLP, dll)"""
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        # Jika ada scaler, normalisasi input terlebih dahulu
        if self.scaler:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        return self.model.predict(X_processed)

class RNNWrapper:
    """Wrapper khusus untuk RNN dengan penanganan dimensi time-steps"""
    def __init__(self, model, scaler, time_steps):
        self.model = model
        self.scaler = scaler
        self.time_steps = time_steps

    def predict(self, X):
        # Normalisasi input
        X_scaled = self.scaler.transform(X)
        
        # Jika input dari simulator frontend (hanya 1 baris statis)
        if len(X_scaled) < self.time_steps:
            # Lakukan padding dengan menduplikasi baris pertama agar memenuhi time_steps
            pad_size = self.time_steps - len(X_scaled)
            X_padded = np.vstack([np.tile(X_scaled[0], (pad_size, 1)), X_scaled])
            X_arr = X_padded.reshape((1, self.time_steps, X_scaled.shape[1]))
        else:
            # Ambil n-baris terakhir sesuai time_steps
            X_arr = X_scaled[-self.time_steps:].reshape((1, self.time_steps, X_scaled.shape[1]))
            
        return self.model.predict(X_arr, verbose=0).flatten()

class SARIMAWrapper:
    """Wrapper untuk SARIMA"""
    def __init__(self, model_fit):
        self.model_fit = model_fit

    def predict(self, X):
        return self.model_fit.forecast(steps=len(X), exog=X).values

# ==========================================
# FUNGSI PENGOLAHAN DATA & PELATIHAN
# ==========================================

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
    
    # Batasi data untuk algoritma yang memakan waktu komputasi tinggi
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
    
    # Inisialisasi Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feat_importance = None
    
    # -----------------------------------------------------
    # PERCABANGAN ALGORITMA MODEL
    # -----------------------------------------------------
    if metode == "SARIMA (Time-Series)":
        # SARIMA bekerja lebih baik dengan data unscaled untuk interpretasi eksogen
        model_sm = SARIMAX(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model_sm.fit(disp=False)
        y_pred = model_fit.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test).values
        model = SARIMAWrapper(model_fit)
        
    elif metode == "Recurrent Neural Network (RNN)":
        time_steps = 3 
        
        # Generator Sliding Window
        def create_sequences(data_X, data_y, ts):
            Xs, ys = [], []
            for i in range(len(data_X) - ts):
                Xs.append(data_X[i:(i + ts)])
                ys.append(data_y.iloc[i + ts])
            return np.array(Xs), np.array(ys)
            
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)
        
        rnn_model = Sequential()
        rnn_model.add(SimpleRNN(32, input_shape=(time_steps, X_train.shape[1]), activation='relu'))
        rnn_model.add(Dense(16, activation='relu'))
        rnn_model.add(Dense(1))
        
        rnn_model.compile(optimizer='adam', loss='mse')
        epochs = parameters.get('epochs', 50)
        rnn_model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=128, verbose=0)
        
        y_pred = rnn_model.predict(X_test_seq, verbose=0).flatten()
        model = RNNWrapper(rnn_model, scaler, time_steps)
        
        # Sesuaikan indeks label aktual dan waktu agar sesuai panjang sekuens prediksi
        y_test = y_test.iloc[time_steps:]
        time_test = time_test.iloc[time_steps:]
        
    elif metode == "XGBoost Regressor":
        raw_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': raw_model.feature_importances_})
        
    elif metode == "Extra Trees Regressor":
        raw_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': raw_model.feature_importances_})
        
    elif metode == "Random Forest":
        raw_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': raw_model.feature_importances_})
        
    elif metode == "Gradient Boosting":
        raw_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': raw_model.feature_importances_})

    elif metode == "AdaBoost Regressor":
        raw_model = AdaBoostRegressor(n_estimators=100, random_state=42)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        feat_importance = pd.DataFrame({'Fitur': features, 'Tingkat Kepentingan': raw_model.feature_importances_})
        
    elif metode == "Multi-Layer Perceptron (MLP)":
        raw_model = MLPRegressor(max_iter=parameters.get('epochs', 200), random_state=42)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        
    else: 
        # Support Vector Regressor (SVR)
        raw_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        raw_model.fit(X_train_scaled, y_train)
        y_pred = raw_model.predict(X_test_scaled)
        model = MLWrapper(raw_model, scaler)
        
    # Pastikan daya tidak negatif secara fisis
    y_pred = np.maximum(y_pred, 0)
        
    # Kalkulasi Metrik Evaluasi
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