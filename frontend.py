import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import database as db
import backend as be
import os

def main():
    st.set_page_config(page_title="Dasbor SDR - Manajemen PLTB", layout="wide")
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
            ["Pilih...", "Dataset Performa Turbin (Fungsi 1)", "Dataset Keandalan Gearbox (Fungsi 2)"]
        )
        
        if pilihan_data != "Pilih...":
            if st.session_state.selected_dataset_name != pilihan_data:
                with st.spinner("Membaca file data..."):
                    loaded_data = be.load_real_data(pilihan_data)
                    
                    if loaded_data is None:
                        st.error("❌ File CSV tidak ditemukan atau format rusak. Pastikan file berada di direktori yang tepat.")
                        st.session_state.active_data = None
                        st.session_state.selected_dataset_name = ""
                    else:
                        st.session_state.active_data = loaded_data
                        st.session_state.selected_dataset_name = pilihan_data
                        # Reset memory model jika ganti dataset
                        for key in list(st.session_state.keys()):
                            if "trained" in key or "hasil" in key:
                                del st.session_state[key]
                            
            if st.session_state.active_data is not None:
                st.success(f"✅ File {pilihan_data} berhasil dimuat!")
                
                tab_data, tab_eda_corr, tab_eda_dist = st.tabs(["Tabel Data", "Matriks Korelasi", "Distribusi Data"])
                
                with tab_data:
                    st.dataframe(st.session_state.active_data.head(1000))
                with tab_eda_corr:
                    df_numeric = st.session_state.active_data.select_dtypes(include=['float64', 'int64'])
                    fig_corr = px.imshow(df_numeric.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
                with tab_eda_dist:
                    # Sample data agar browser tidak berat saat plotting boxplot
                    df_sample = df_numeric.sample(n=min(5000, len(df_numeric)), random_state=42)
                    fig_box = px.box(df_sample.melt(), x="variable", y="value", color="variable", title="Sebaran Data & Outlier")
                    st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================
    # HALAMAN 2: FUNGSI 1 (REGRESI DAYA)
    # ==========================================
    elif menu == "2. Fungsi 1 (Prediksi Daya Turbin)":
        st.header("Modul Fungsi 1: Prediksi Daya Output Turbin (Regresi)")

        if st.session_state.active_data is None or "Performa Turbin" not in st.session_state.selected_dataset_name:
            st.warning("⚠️ Silakan muat 'Dataset Performa Turbin' terlebih dahulu di Menu 1!")
            return

        st.subheader("⚙️ 1. Konfigurasi Model Lanjutan")
        col_method, col_setup = st.columns(2)

        with col_method:
            metode = st.selectbox("Algoritma Prediksi / Time-Series:", [
                "Random Forest", "Recurrent Neural Network (RNN)", "SARIMA (Time-Series)", 
                "XGBoost Regressor", "Extra Trees Regressor", "Gradient Boosting", 
                "AdaBoost Regressor", "Support Vector Regressor (SVR)", "Multi-Layer Perceptron (MLP)"
            ])
            st.session_state["f1_metode"] = metode

        with col_setup:
            train_size = st.slider("Ukuran Data Latih (%)", 20, 95, 80, 5, key="f1_slider")
            epochs = st.number_input("Max Epochs (Khusus RNN & MLP)", 10, 500, 50, key="f1_epoch")
            st.session_state["f1_setup"] = {"train_size": train_size, "epochs": epochs}

        if st.button("🚀 Mulai Pelatihan Machine Learning", use_container_width=True, key="f1_btn"):
            with st.spinner(f"Melatih {st.session_state['f1_metode']}..."):
                hasil_training = be.run_regression_fungsi1(st.session_state["f1_metode"], st.session_state["f1_setup"], st.session_state.active_data)
                db.log_training("Fungsi 1 (Regresi)", st.session_state["f1_metode"], st.session_state["f1_setup"], f"Sukses ({hasil_training['metrik1_nilai']})")
                
            st.success("Training Selesai!")
            st.session_state["f1_hasil"] = hasil_training
            st.session_state["f1_trained"] = True

        st.markdown("---")

        if st.session_state.get("f1_trained"):
            hasil = st.session_state["f1_hasil"]
            st.subheader("📊 2. Evaluasi Kinerja (Regresi)")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric(hasil["metrik1_nama"], hasil["metrik1_nilai"])
            col_m2.metric(hasil["metrik2_nama"], hasil["metrik2_nilai"])
            col_m3.metric(hasil["metrik3_nama"], hasil["metrik3_nilai"])
            col_m4.metric(hasil["metrik4_nama"], hasil["metrik4_nilai"])
            
            tab_line, tab_scatter, tab_feat = st.tabs(["Tren Waktu", "Aktual vs Prediksi", "Feature Importance"])
            with tab_line:
                fig1 = px.line(hasil["chart_data"], x='Waktu', y=['Daya Aktual', 'Prediksi Daya'])
                st.plotly_chart(fig1, use_container_width=True)
            with tab_scatter:
                fig_scatter = px.scatter(hasil["chart_data"], x="Daya Aktual", y="Prediksi Daya", opacity=0.6)
                st.plotly_chart(fig_scatter, use_container_width=True)
            with tab_feat:
                if hasil["feature_importance"] is not None:
                    fig3 = px.bar(hasil["feature_importance"], x="Tingkat Kepentingan", y="Fitur", orientation='h')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Algoritma ini tidak menyediakan metrik Feature Importance secara eksplisit.")

            st.markdown("---")
            st.subheader("🎮 3. Control Room Simulator (Fungsi 1)")
            
            if st.session_state["f1_metode"] == "SARIMA (Time-Series)":
                st.info("ℹ️ Simulator statis dinonaktifkan untuk SARIMA karena arsitekturnya membutuhkan sekuens waktu historis berkelanjutan.")
            else:
                model = hasil["model"]
                col_slider, col_gauge = st.columns([1, 2])
                with col_slider:
                    ws_10 = st.slider("Kecepatan Angin 10m (m/s)", 0.0, 25.0, 6.0)
                    ws_100 = st.slider("Kecepatan Angin 100m (m/s)", 0.0, 30.0, 8.5)
                    wd_100 = st.slider("Arah Angin (Derajat)", 0, 360, 180) 
                    temp_2 = st.slider("Suhu Udara (°C)", 0.0, 40.0, 25.0)
                    rh_2 = st.slider("Kelembaban (%)", 0.0, 100.0, 80.0)
                    
                    input_data = pd.DataFrame([[ws_10, ws_100, wd_100, temp_2, rh_2]], columns=['WS_10m', 'WS_100m', 'WD_100m', 'Temp_2m', 'RelHum_2m'])
                    prediksi_mw = max(0, model.predict(input_data)[0])
                
                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = prediksi_mw, title = {'text': "⚡ Estimasi Daya (MW)"},
                        gauge = {'axis': {'range': [None, 3]}, 'bar': {'color': "#1f77b4"}}
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)

    # ==========================================
    # HALAMAN 3: FUNGSI 2 (KLASIFIKASI ANOMALI)
    # ==========================================
    elif menu == "3. Fungsi 2 (Deteksi Anomali Gearbox)":
        st.header("Modul Fungsi 2: Deteksi Anomali Gearbox (Klasifikasi)")

        if st.session_state.active_data is None or "Keandalan Gearbox" not in st.session_state.selected_dataset_name:
            st.warning("⚠️ Silakan muat 'Dataset Keandalan Gearbox' terlebih dahulu di Menu 1!")
            return

        st.subheader("⚙️ 1. Konfigurasi Model Klasifikasi")
        col_method, col_setup = st.columns(2)

        with col_method:
            metode = st.selectbox("Algoritma Klasifikasi:", [
                "Random Forest Classifier", "Support Vector Classifier (SVC)", "K-Nearest Neighbors (KNN)"
            ])
            st.session_state["f2_metode"] = metode

        with col_setup:
            train_size = st.slider("Ukuran Data Latih (%)", 20, 95, 80, 5, key="f2_slider")
            st.session_state["f2_setup"] = {"train_size": train_size}

        if st.button("🚀 Mulai Pelatihan Klasifikasi", use_container_width=True, key="f2_btn"):
            with st.spinner(f"Melatih {st.session_state['f2_metode']}..."):
                hasil_training = be.run_anomaly_fungsi2(st.session_state["f2_metode"], st.session_state["f2_setup"], st.session_state.active_data)
                db.log_training("Fungsi 2 (Klasifikasi)", st.session_state["f2_metode"], st.session_state["f2_setup"], f"Sukses ({hasil_training['metrik1_nilai']})")
                
            st.success("Training Klasifikasi Selesai!")
            st.session_state["f2_hasil"] = hasil_training
            st.session_state["f2_trained"] = True

        st.markdown("---")

        if st.session_state.get("f2_trained"):
            hasil = st.session_state["f2_hasil"]
            st.subheader("📊 2. Evaluasi Kinerja (Klasifikasi)")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric(hasil["metrik1_nama"], hasil["metrik1_nilai"])
            col_m2.metric(hasil["metrik2_nama"], hasil["metrik2_nilai"])
            col_m3.metric(hasil["metrik3_nama"], hasil["metrik3_nilai"])
            col_m4.metric(hasil["metrik4_nama"], hasil["metrik4_nilai"])
            
            tab_cm, tab_feat = st.tabs(["Confusion Matrix", "Feature Importance"])
            with tab_cm:
                cm_df = hasil["cm_data"]
                # Menggunakan plotly figure factory untuk Heatmap teranotasi (Confusion Matrix)
                fig_cm = ff.create_annotated_heatmap(
                    z=cm_df.values, x=cm_df.columns.tolist(), y=cm_df.index.tolist(), 
                    colorscale='Blues', showscale=True
                )
                fig_cm.update_layout(title_text='Matriks Kebingungan (Confusion Matrix)', title_x=0.5)
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with tab_feat:
                if hasil["feature_importance"] is not None:
                    fig_feat = px.bar(hasil["feature_importance"], x="Tingkat Kepentingan", y="Fitur", orientation='h')
                    st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info("Algoritma ini (SVC/KNN) tidak mengekstrak Feature Importance eksplisit seperti algoritma Tree.")

            st.markdown("---")
            # Simulator Predictive Maintenance (Fungsi 2)
            st.subheader("🔍 3. Simulator Predictive Maintenance (Gearbox)")
            model = hasil["model"]
            
            col_sim1, col_sim2, col_status = st.columns([1, 1, 1.5])
            with col_sim1:
                oil_temp = st.number_input("Suhu Oli (°C)", value=60.0, step=0.5)
                bearing_temp = st.number_input("Suhu Bantalan (°C)", value=67.0, step=0.5)
                oil_press = st.number_input("Tekanan Oli (Bar)", value=4.4, step=0.1)
                particle = st.number_input("Jumlah Partikel (ppm)", value=120, step=10)
            with col_sim2:
                vib_x = st.number_input("Vibrasi X (mm/s)", value=0.012, step=0.005, format="%.4f")
                vib_y = st.number_input("Vibrasi Y (mm/s)", value=0.011, step=0.005, format="%.4f")
                vib_z = st.number_input("Vibrasi Z (mm/s)", value=0.014, step=0.005, format="%.4f")
            
            input_anomali = pd.DataFrame([[oil_temp, bearing_temp, vib_x, vib_y, vib_z, oil_press, particle]], 
                                         columns=['gearbox_oil_temp', 'gearbox_bearing_temp', 'vibration_x', 'vibration_y', 'vibration_z', 'oil_pressure', 'particle_count'])
            
            try:
                prediksi_kelas = model.predict(input_anomali)[0]
            except Exception as e:
                st.error(f"Error inferensi: {e}")
                prediksi_kelas = 0

            with col_status:
                st.markdown("#### Status Keandalan Mesin")
                if prediksi_kelas == 0:
                    st.success("🟢 **NORMAL** \n\nParameter fisis beroperasi dalam batas aman. Tidak ada indikasi kerusakan struktural atau pelumasan.")
                else:
                    st.error("🔴 **ANOMALI TERDETEKSI (PERINGATAN!)** \n\nTerdeteksi pergeseran parameter fisis dari profil normal. Jadwalkan inspeksi *gearbox* segera untuk mencegah kegagalan fatal.")

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
            st.info("Database eksperimen masih kosong.")

if __name__ == "__main__":
    main()