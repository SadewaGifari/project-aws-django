# prediksi/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import json
import os
from django.conf import settings
import numpy as np
import pandas as pd # Kita butuh pandas untuk olah data
import joblib
from proyekjamur import settings

# --- TAHAP INTEGRASI: Import library InfluxDB ---
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
# ----------------------------------------------------


# --- TAHAP INTEGRASI: Konfigurasi Koneksi InfluxDB ---
INFLUX_URL = "http://103.151.63.81:8087"
INFLUX_TOKEN = "Sv2J_33XCeYy4SgQSijT09qr4OTjrUjcLz59oJci2nny46OzPuclSy3R3CIFe3-PJuChMogeiwuVL7iRnjM8Mg=="
INFLUX_ORG = "proyek_jamur"
INFLUX_BUCKET = "sensor_data"

# Membuat koneksi client ke InfluxDB
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = influx_client.query_api()
# ----------------------------------------------------


# --- Logika Model AI (TIDAK BERUBAH) ---
MODEL_DIR = os.path.join(settings.BASE_DIR, 'prediksi', 'ml_model')
AI_MODEL = joblib.load(os.path.join(MODEL_DIR, 'sensor_model.pkl'))
LABEL_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'sensor_label_encoder.pkl'))
VENT_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'ventilation_encoder.pkl'))
LIGHT_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'light_encoder.pkl'))

def prediksi_risiko(suhu, kelembapan):
    try:
        ph_asumsi = 7.0
        ventilasi_asumsi = 'low'
        cahaya_asumsi = 'low'
        ventilasi_enc = VENT_ENCODER.transform([ventilasi_asumsi])[0]
        cahaya_enc = LIGHT_ENCODER.transform([cahaya_asumsi])[0]
        fitur_input = np.array([[suhu, kelembapan, ph_asumsi, ventilasi_enc, cahaya_enc]])
        hasil_prediksi_numerik = AI_MODEL.predict(fitur_input)
        risiko_teks = LABEL_ENCODER.inverse_transform(hasil_prediksi_numerik)[0]
        if risiko_teks == 'high':
            return "Tinggi", "Model AI (5 Fitur) mendeteksi kondisi ideal untuk pertumbuhan jamur."
        else:
            return "Aman", "Model AI (5 Fitur) memprediksi kondisi saat ini tidak mendukung pertumbuhan jamur."
    except Exception as e:
        return "Error", f"Terjadi kesalahan saat prediksi AI: {e}"
# ----------------------------------------------------

def dashboard_prediksi(request):
    data_sensor = []

    if settings.query_api:  # pastikan koneksi Influx tersedia
        try:
            query = 'from(bucket:"jamur") |> range(start: -1h)'
            tables = settings.query_api.query(query)
            for table in tables:
                for record in table.records:
                    data_sensor.append({
                        "waktu": record.get_time(),
                        "suhu": record.get_value()
                    })
        except Exception as e:
            print(f"[WARNING] Gagal ambil data Influx: {e}")
    else:
        print("⚠️ InfluxDB tidak aktif — tampilkan dummy data.")



# --- TULIS ULANG FUNGSI DASHBOARD ---
@login_required
def dashboard_prediksi(request):
    # Query Flux untuk mengambil data terakhir
    flux_query_latest = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -1d) 
        |> filter(fn: (r) => r["_measurement"] == "lingkungan")
        |> filter(fn: (r) => r["_field"] == "humidity" or r["_field"] == "temperature")
        |> last()
    '''
    tables_latest = query_api.query(flux_query_latest, org=INFLUX_ORG)

    suhu, kelembapan, timestamp = 0, 0, "Belum ada data"
    if tables_latest:
        for table in tables_latest:
            for record in table.records:
                if record.get_field() == 'temperature':
                    suhu = record.get_value()
                elif record.get_field() == 'humidity':
                    kelembapan = record.get_value()
        timestamp = tables_latest[0].records[0].get_time()

    level_risiko, rekomendasi = prediksi_risiko(suhu, kelembapan)

    # Query Flux untuk data chart 12 jam terakhir
    flux_query_chart = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -12h)
        |> filter(fn: (r) => r["_measurement"] == "lingkungan")
        |> filter(fn: (r) => r["_field"] == "humidity" or r["_field"] == "temperature")
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
    '''
    df_chart = query_api.query_data_frame(flux_query_chart, org=INFLUX_ORG)

    labels, suhu_data, kelembapan_data = [], [], []
    if not df_chart.empty:
        df_chart_temp = df_chart[df_chart['_field'] == 'temperature']
        df_chart_hum = df_chart[df_chart['_field'] == 'humidity']
        labels = df_chart_temp['_time'].dt.strftime('%H:%M').tolist()
        suhu_data = df_chart_temp['_value'].tolist()
        kelembapan_data = df_chart_hum['_value'].tolist()

    context = {
        'suhu': suhu, 'kelembapan': kelembapan, 'waktu_update': timestamp,
        'level_risiko': level_risiko, 'rekomendasi': rekomendasi,
        'chart_labels': json.dumps(labels), 'chart_suhu_data': json.dumps(suhu_data),
        'chart_kelembapan_data': json.dumps(kelembapan_data),
    }
    return render(request, 'prediksi/dashboard.html', context)


# --- TULIS ULANG FUNGSI LAPORAN ---
@login_required
def laporan_historis(request):
    days_to_filter = int(request.GET.get('days', 7))
    
    # Query Flux untuk data historis dan statistik
    flux_query_report = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -{days_to_filter}d)
        |> filter(fn: (r) => r["_measurement"] == "lingkungan")
        |> filter(fn: (r) => r["_field"] == "humidity" or r["_field"] == "temperature")
    '''
    df_report = query_api.query_data_frame(flux_query_report, org=INFLUX_ORG)

    stats = {}
    page_obj = [] # Untuk saat ini, kita sederhanakan tanpa paginasi dari InfluxDB
    
    if not df_report.empty:
        df_temp = df_report[df_report['_field'] == 'temperature']['_value']
        df_hum = df_report[df_report['_field'] == 'humidity']['_value']
        stats = {
            'avg_suhu': df_temp.mean(), 'max_suhu': df_temp.max(), 'min_suhu': df_temp.min(),
            'avg_kelembapan': df_hum.mean(), 'max_kelembapan': df_hum.max(), 'min_kelembapan': df_hum.min(),
        }
        # Mengambil 100 data terakhir untuk ditampilkan di tabel
        page_obj = df_report.sort_values(by='_time', ascending=False).head(100)
        # Mengubah format data untuk template
        page_obj = page_obj.pivot(index='_time', columns='_field', values='_value').reset_index()
        page_obj = page_obj.sort_values(by='_time', ascending=False).to_dict('records')


    # Data untuk chart
    labels, suhu_data, kelembapan_data = [], [], []
    if not df_report.empty:
        df_chart = df_report.sort_values(by='_time')
        df_chart_temp = df_chart[df_chart['_field'] == 'temperature']
        df_chart_hum = df_chart[df_chart['_field'] == 'humidity']
        labels = df_chart_temp['_time'].dt.strftime('%d %b %H:%M').tolist()
        suhu_data = df_chart_temp['_value'].tolist()
        kelembapan_data = df_chart_hum['_value'].tolist()

    context = {
        'page_obj': page_obj, 'stats': stats, 'days_filtered': days_to_filter,
        'chart_labels': json.dumps(labels), 'chart_suhu_data': json.dumps(suhu_data),
        'chart_kelembapan_data': json.dumps(kelembapan_data),
    }
    return render(request, 'prediksi/laporan.html', context)


# --- HAPUS FUNGSI LAMA ---
# Fungsi @csrf_exempt simpan_data_sensor(request) sudah tidak diperlukan lagi
# karena Node-RED yang sekarang menangani penyimpanan data. Anda bisa menghapusnya.