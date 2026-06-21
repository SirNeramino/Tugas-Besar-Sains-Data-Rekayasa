import sqlite3
import pandas as pd
import json
from datetime import datetime

# Nama file database SQLite yang akan otomatis terbuat di foldermu
DB_NAME = 'eksperimen_model.db'

def init_db():
    """Membuat tabel logs jika belum ada di database saat aplikasi pertama kali dijalankan."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT,
            fungsi TEXT,
            metode TEXT,
            parameter TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_training(fungsi, metode, setup, status):
    """Menyimpan riwayat (log) training model ke dalam database SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Ambil stempel waktu saat ini
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Konversi parameter (ukuran train/test, epochs, dll) ke string JSON agar bisa disimpan
    setup_str = json.dumps(setup) if isinstance(setup, dict) else str(setup)
    
    cursor.execute('''
        INSERT INTO logs (waktu, fungsi, metode, parameter, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (waktu_sekarang, fungsi, metode, setup_str, status))
    
    conn.commit()
    conn.close()

def get_all_logs():
    """Mengambil seluruh data riwayat eksperimen untuk ditampilkan di Halaman 4 Dasbor."""
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT waktu, fungsi, metode, parameter, status FROM logs ORDER BY waktu DESC"
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        # Kembalikan dataframe kosong jika terjadi error (misal tabel belum ada isinya)
        df = pd.DataFrame() 
        
    conn.close()
    return df