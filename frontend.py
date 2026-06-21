import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import database as db
import backend as be
import os

def main():
    st.set_page_config(page_title="Dasbor SDR - Analisis PLTB", layout="wide")
    db.init_db()

    logo_path = "Group-48095824-1.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        st.sidebar.markdown("### ⚙️ SAINS DATA REKAYASA")
        
    st.sidebar.title("Navigasi Utama")
    menu = st.sidebar.radio("Pilih Halaman", [
        "1. Load Dataset & EDA", 
        "2. Fungsi 1 (Prediksi Daya Turbin)", 
        "3. Fungsi 2 (Deteksi Anomali Gearbox)", 
        "4. Laporan Database (Log)"
    ])

    if "active_data" not in st.session_state:
        st.session_state.active_data = None
    if "selected_dataset_name" not in st.session_state:
        st.session_state.selected_dataset_name = ""

    # ==========================================
    # HALAMAN 1: LOAD DATASET & EDA
    # ==========================================
    if menu == "1. Load Dataset & EDA":
        st.header("Modul 1: Load Data & Exploratory Data Analysis (EDA)")
        pilihan_data = st.selectbox(
            "Pilih Dataset Analisis:", 
            ["Pilih...", "Dataset Performa Turbin (turbine_speed_direction.csv)", "Dataset Keandalan Gearbox (turbine_gearbox.csv)"]
        )
        
        if pilihan_data != "Pilih...":
            if st.session_state.selected_dataset_name != pilihan_data:
                with st.spinner("Membaca file data dari penyimpanan..."):
                    loaded_data = be.load_real_data(pilihan_data)
                    
                    # Tambahan Error Handling jika file tidak ditemukan/rusak
                    if loaded_data is None:
                        st.error("❌ File CSV tidak ditemukan atau format rusak. Pastikan file berada di direktori yang tepat.")
                        st.session_state.active_data = None
                        st.session_state.selected_dataset_name = ""
                    else:
                        st.session_state.active_data = loaded_data
                        st.session_state.selected_dataset_name = pilihan_data
                        # Reset model jika ganti dataset
                        for key in list(st.session_state.keys()):
                            if "trained" in key or "hasil" in key:
                                del st.session_state[key]
                            
            if st.session_state.active_data is not None:
                st.success("✅ File berhasil dimuat!")
                
                if "Performa Turbin" in pilihan_data:
                    tab_data, tab_eda_corr, tab_eda_dist, tab_windrose = st.tabs(["Tabel Data Fisis", "Matriks Korelasi", "Distribusi Data", "Wind Rose (Arah Angin)"])
                    with tab_windrose:
                        st.subheader("Visualisasi Polar Arah & Kecepatan Angin (Wind Rose)")
                        df_wind = st.session_state.active_data.sample(n=min(10000, len(st.session_state.active_data)), random_state=42)
                        fig_polar = px.scatter_polar(df_wind, r="WS_100m", theta="WD_100m", color="Power",
                                                     color_continuous_scale="Turbo", template="plotly_dark",
                                                     title="Profil Arah Angin di Ketinggian Hub (100m)")
                        st.plotly_chart(fig_polar, use_container_width=True)
                else:
                    tab_data, tab_eda_corr, tab_eda_dist = st.tabs(["Tabel Data Fisis", "Matriks Korelasi", "Distribusi Data"])
                
                with tab_data:
                    st.dataframe(st.session_state.active_data.head(1000))
                with tab_eda_corr:
                    df_numeric = st.session_state.active_data.select_dtypes(include=['float64', 'int64'])
                    fig_corr = px.imshow(df_numeric.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
                with tab_eda_dist:
                    fig_box = px.box(df_numeric.melt(), x="variable", y="value", color="variable", title="Sebaran Data (Identifikasi Outlier Fisis)")
                    st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================
    # HALAMAN 2 & 3: MACHINE LEARNING
    # ==========================================
    elif menu in ["2. Fungsi 1 (Prediksi Daya Turbin)", "3. Fungsi 2 (Deteksi Anomali Gearbox)"]:
        is_fungsi_1 = "Fungsi 1" in menu
        nama_fungsi = "Fungsi 1" if is_fungsi_1 else "Fungsi 2"
        
        st.header(f"Modul: {menu}")
        
        # --- BLOK KHUSUS: FUNGSI 2 DIKUNCI SEMENTARA ---
        if not is_fungsi_1:
            st.warning("🚧 **PEMELIHARAAN SISTEM:** Modul Fungsi 2 (Deteksi Anomali) belum diaktifkan pada iterasi ini. Silakan gunakan Fungsi 1 terlebih dahulu.")
            return # Menghentikan eksekusi kode di bawahnya untuk Fungsi 2
        # --------------------------------------------------

        if st.session_state.active_data is None:
            st.warning("⚠️ Silakan Load Dataset terlebih dahulu di Menu '1. Load Dataset & EDA'!")
            return

        if "Performa Turbin" not in st.session_state.selected_dataset_name:
            st.error("❌ Wajib menggunakan 'Dataset Performa Turbin' untuk menjalankan Fungsi 1.")
            return

        st.subheader("⚙️ 1. Konfigurasi Model Lanjutan")
        col_method, col_setup = st.columns(2)

        with col_method:
            metode = st.selectbox("Algoritma Prediksi / Time-Series:", [
                "Recurrent Neural Network (RNN)", "SARIMA (Time-Series)", "XGBoost Regressor", 
                "Extra Trees Regressor", "Random Forest", "Gradient Boosting", 
                "AdaBoost Regressor", "Support Vector Regressor (SVR)", "Multi-Layer Perceptron (MLP)"
            ])
            st.session_state[f"{nama_fungsi}_metode"] = metode

        with col_setup:
            train_size = st.slider("Ukuran Data Latih (%)", min_value=20, max_value=95, value=80, step=5)
            epochs = st.number_input("Max Epochs (Khusus RNN & MLP)", 10, 500, 50)
            st.session_state[f"{nama_fungsi}_setup"] = {"train_size": train_size, "epochs": epochs}

        if st.button("🚀 Mulai Pelatihan Machine Learning", use_container_width=True):
            with st.spinner(f"Melatih {st.session_state.get(f'{nama_fungsi}_metode')}... (Ini mungkin memakan waktu)"):
                hasil_training = be.run_regression_fungsi1(st.session_state[f"{nama_fungsi}_metode"], st.session_state[f"{nama_fungsi}_setup"], st.session_state.active_data)
                
                db.log_training(nama_fungsi, st.session_state[f"{nama_fungsi}_metode"], st.session_state[f"{nama_fungsi}_setup"], f"Sukses ({hasil_training['metrik1_nilai']})")
                
            st.success("Training Selesai! Silakan lihat hasil evaluasi visual di bawah.")
            st.session_state[f"{nama_fungsi}_hasil"] = hasil_training
            st.session_state[f"{nama_fungsi}_trained"] = True

        st.markdown("---")

        if st.session_state.get(f"{nama_fungsi}_trained"):
            st.subheader("📊 2. Evaluasi Kinerja (4 Metrik Utama)")
            hasil = st.session_state[f"{nama_fungsi}_hasil"]
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric(label=hasil["metrik1_nama"], value=hasil["metrik1_nilai"])
            col_m2.metric(label=hasil["metrik2_nama"], value=hasil["metrik2_nilai"])
            col_m3.metric(label=hasil["metrik3_nama"], value=hasil["metrik3_nilai"])
            col_m4.metric(label=hasil["metrik4_nama"], value=hasil["metrik4_nilai"])
            
            tab_line, tab_scatter_corr, tab_resid, tab_feat = st.tabs(["Tren Waktu (Line)", "Korelasi Model (Scatter)", "Distribusi Galat", "Feature Importance"])
            with tab_line:
                fig1 = px.line(hasil["chart_data"], x='Waktu', y=['Daya Aktual', 'Prediksi Daya'], title="Tren Perbandingan Terhadap Waktu")
                fig1.update_layout(xaxis_title="Waktu (Tanggal & Jam)", yaxis_title="Daya (MW)", hovermode="x unified")
                st.plotly_chart(fig1, use_container_width=True)
            with tab_scatter_corr:
                fig_scatter_reg = px.scatter(hasil["chart_data"], x="Daya Aktual", y="Prediksi Daya", opacity=0.6, color_discrete_sequence=['#1f77b4'])
                min_val = min(hasil["chart_data"]["Daya Aktual"].min(), hasil["chart_data"]["Prediksi Daya"].min())
                max_val = max(hasil["chart_data"]["Daya Aktual"].max(), hasil["chart_data"]["Prediksi Daya"].max())
                fig_scatter_reg.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
                st.plotly_chart(fig_scatter_reg, use_container_width=True)
            with tab_resid:
                fig2 = px.histogram(hasil["chart_data"], x="Galat (Residual)", nbins=50, marginal="box", color_discrete_sequence=['indianred'])
                st.plotly_chart(fig2, use_container_width=True)
            with tab_feat:
                if hasil["feature_importance"] is not None:
                    fig3 = px.bar(hasil["feature_importance"], x="Tingkat Kepentingan", y="Fitur", orientation='h', color="Tingkat Kepentingan", color_continuous_scale='viridis')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Algoritma Time-Series (RNN/SARIMA) atau MLP memproses fitur dalam dimensi yang tidak bisa diekstrak 'Feature Importance' statisnya secara eksplisit seperti pada algoritma berbasis Tree.")

            st.markdown("---")
            
            # ==========================================
            # 3. SIMULATOR & ANALISIS BISNIS
            # ==========================================
            st.subheader("🎮 3. Control Room Simulator & Analisis Bisnis")
            
            # PENAMBAHAN LOGIKA PROTEKSI SARIMA
            if st.session_state[f"{nama_fungsi}_metode"] == "SARIMA (Time-Series)":
                st.info("ℹ️ **Simulator Control Room Dinonaktifkan.** \n\nModel SARIMA adalah model murni deret waktu (*time-series*) yang membutuhkan urutan data historis untuk memprediksi nilai masa depan. Memberikan satu baris input cuaca statis (dari slider) tidak sesuai dengan arsitektur matematis SARIMA.")
            
            else:
                model = hasil["model"]
                
                col_slider, col_gauge = st.columns([1, 2])
                
                with col_slider:
                    st.markdown("**Panel Kendali Cuaca**")
                    ws_100 = st.slider("Kecepatan Angin Hub 100m (m/s)", 0.0, 30.0, 8.5, 0.5)
                    ws_10 = st.slider("Kecepatan Angin Permukaan 10m (m/s)", 0.0, 25.0, 6.0, 0.5)
                    wd_100 = st.slider("Arah Angin (Derajat)", 0, 360, 180, 5) 
                    temp_2 = st.slider("Suhu Udara (°C)", 0.0, 40.0, 25.0, 0.5)
                    rh_2 = st.slider("Kelembaban Relatif (%)", 0.0, 100.0, 80.0, 1.0)
                    
                    input_data = pd.DataFrame([[ws_10, ws_100, wd_100, temp_2, rh_2]], columns=['WS_10m', 'WS_100m', 'WD_100m', 'Temp_2m', 'RelHum_2m'])
                    
                    try:
                        prediksi_mw = model.predict(input_data)[0]
                        prediksi_mw = max(0, prediksi_mw)
                    except Exception as e:
                        st.error(f"Error pada inferensi model: {e}")
                        prediksi_mw = 0
                
                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediksi_mw,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "⚡ Estimasi Daya Output (Megawatt)", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [None, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#1f77b4"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 1], 'color': '#ffcccb'},
                                {'range': [1, 2], 'color': '#ffffcc'},
                                {'range': [2, 3], 'color': '#ccffcc'}]
                        }))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("#### 📈 Proyeksi Dampak Operasional & Finansial")
                col_b1, col_b2, col_b3 = st.columns(3)
                
                kw_output = prediksi_mw * 1000
                rumah_tercakup = int(kw_output / 0.9)
                pendapatan_per_hari = kw_output * 24 * 1444
                
                col_b1.info(f"**🏠 Cakupan Listrik:**\n\nMenyuplai **{rumah_tercakup:,}** Rumah Tangga (Asumsi Daya 900VA)")
                col_b2.success(f"**💰 Proyeksi Pendapatan:**\n\n**Rp {pendapatan_per_hari:,.0f}** / Hari (Asumsi Tarif Rp1.444/kWh PLN)")
                col_b3.warning(f"**🌱 Karbon Terhindar:**\n\n**{(kw_output * 24 * 0.85) / 1000:.2f} Ton CO2** / Hari (vs PLTU Batu Bara)")

    # ==========================================
    # HALAMAN 4: LAPORAN DATABASE
    # ==========================================
    elif menu == "4. Laporan Database (Log)":
        st.header("Modul 4: Dasbor Histori Eksperimen Model")
        df_logs = db.get_all_logs()
        if not df_logs.empty:
            st.subheader("Frekuensi Penggunaan Algoritma")
            fig_bar = px.bar(df_logs['metode'].value_counts().reset_index(), x='count', y='metode', orientation='h', color='count')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.subheader("Tabel Riwayat Eksperimen ML")
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("Database masih kosong.")

if __name__ == "__main__":
    main()