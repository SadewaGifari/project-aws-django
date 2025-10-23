# prediksi/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import DataSensor
from django.contrib.auth.decorators import login_required
from django.utils import timezone
import datetime
from django.db.models import Avg, Max, Min
from django.core.paginator import Paginator

# --- TAHAP INTEGRASI: Import library yang dibutuhkan ---
import joblib
import os
from django.conf import settings
import numpy as np
# ----------------------------------------------------


# === TAHAP INTEGRASI: Muat SEMUA Model & Encoder ===
MODEL_DIR = os.path.join(settings.BASE_DIR, 'prediksi', 'ml_model')

# Coba load model & encoder dengan proteksi error
try:
    AI_MODEL = joblib.load(os.path.join(MODEL_DIR, 'sensor_model.pkl'))
    LABEL_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'sensor_label_encoder.pkl'))
    VENT_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'ventilation_encoder.pkl'))
    LIGHT_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'light_encoder.pkl'))
    print("[INFO] Model ML dan encoder berhasil dimuat.")
except Exception as e:
    AI_MODEL = LABEL_ENCODER = VENT_ENCODER = LIGHT_ENCODER = None
    print(f"[WARNING] Gagal memuat model ML: {e}")
# ----------------------------------------------------


# === Fungsi Prediksi Risiko dengan AI ===
def prediksi_risiko(suhu, kelembapan):
    """
    Fungsi ini menggunakan model AI 5-fitur yang sudah dilatih.
    """
    # Cek apakah model dan encoder sudah siap
    if not AI_MODEL or not LABEL_ENCODER:
        return "Error", "Model AI belum dimuat. Pastikan file .pkl tersedia di folder ml_model."

    try:
        # Nilai asumsi untuk fitur tambahan
        ph_asumsi = 7.0
        ventilasi_asumsi = 'low'
        cahaya_asumsi = 'low'

        # Encoding fitur kategorikal
        ventilasi_enc = VENT_ENCODER.transform([ventilasi_asumsi])[0]
        cahaya_enc = LIGHT_ENCODER.transform([cahaya_asumsi])[0]

        # Siapkan 5 fitur input (urutan harus sama seperti di Colab)
        fitur_input = np.array([[suhu, kelembapan, ph_asumsi, ventilasi_enc, cahaya_enc]])

        # Prediksi
        hasil_prediksi_numerik = AI_MODEL.predict(fitur_input)

        # Konversi hasil ke label teks
        risiko_teks = LABEL_ENCODER.inverse_transform(hasil_prediksi_numerik)[0]

        # Rekomendasi
        if risiko_teks.lower() == 'high':
            return "Tinggi", "Model AI mendeteksi kondisi ideal untuk pertumbuhan jamur."
        else:
            return "Aman", "Model AI memprediksi kondisi saat ini tidak mendukung pertumbuhan jamur."

    except Exception as e:
        return "Error", f"Terjadi kesalahan saat prediksi AI: {e}"


# === Endpoint untuk menyimpan data sensor ===
@csrf_exempt
def simpan_data_sensor(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            temperature = data.get('temperature')
            humidity = data.get('humidity')

            if temperature is not None and humidity is not None:
                DataSensor.objects.create(temperature=temperature, humidity=humidity)
                return JsonResponse({'status': 'sukses', 'message': 'Data berhasil disimpan'}, status=201)
            else:
                return JsonResponse({'status': 'gagal', 'message': 'Data `temperature` atau `humidity` tidak lengkap'}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({'status': 'gagal', 'message': 'Format JSON salah'}, status=400)

        except Exception as e:
            return JsonResponse({'status': 'gagal', 'message': f'Kesalahan server: {e}'}, status=500)

    return JsonResponse({'status': 'gagal', 'message': 'Metode tidak diizinkan'}, status=405)


# === Dashboard Prediksi ===
@login_required
def dashboard_prediksi(request):
    data_terakhir = DataSensor.objects.order_by('-timestamp').first()

    if data_terakhir:
        suhu = data_terakhir.temperature
        kelembapan = data_terakhir.humidity
        timestamp = data_terakhir.timestamp
        level_risiko, rekomendasi = prediksi_risiko(suhu, kelembapan)
    else:
        suhu, kelembapan, timestamp = (0, 0, "Belum ada data")
        level_risiko, rekomendasi = ("Tidak Diketahui", "Belum ada data sensor untuk dianalisis.")

    # Ambil data 12 jam terakhir untuk grafik
    twelve_hours_ago = timezone.now() - datetime.timedelta(hours=12)
    data_historis = DataSensor.objects.filter(timestamp__gte=twelve_hours_ago).order_by('timestamp')

    labels = [d.timestamp.strftime('%H:%M') for d in data_historis]
    suhu_data = [d.temperature for d in data_historis]
    kelembapan_data = [d.humidity for d in data_historis]

    context = {
        'suhu': suhu,
        'kelembapan': kelembapan,
        'waktu_update': timestamp,
        'level_risiko': level_risiko,
        'rekomendasi': rekomendasi,
        'chart_labels': json.dumps(labels),
        'chart_suhu_data': json.dumps(suhu_data),
        'chart_kelembapan_data': json.dumps(kelembapan_data),
    }

    return render(request, 'prediksi/dashboard.html', context)


# === Laporan Historis ===
@login_required
def laporan_historis(request):
    try:
        days_to_filter = int(request.GET.get('days', 7))
    except ValueError:
        days_to_filter = 7

    start_date = timezone.now() - datetime.timedelta(days=days_to_filter)
    data_list = DataSensor.objects.filter(timestamp__gte=start_date).order_by('-timestamp')

    # Statistik agregat
    stats = data_list.aggregate(
        avg_suhu=Avg('temperature'),
        max_suhu=Max('temperature'),
        min_suhu=Min('temperature'),
        avg_kelembapan=Avg('humidity'),
        max_kelembapan=Max('humidity'),
        min_kelembapan=Min('humidity'),
    )

    # Pagination
    paginator = Paginator(data_list, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    # Data untuk chart
    data_for_chart = data_list.order_by('timestamp')
    labels = [d.timestamp.strftime('%d %b %H:%M') for d in data_for_chart]
    suhu_data = [d.temperature for d in data_for_chart]
    kelembapan_data = [d.humidity for d in data_for_chart]

    context = {
        'page_obj': page_obj,
        'stats': stats,
        'days_filtered': days_to_filter,
        'chart_labels': json.dumps(labels),
        'chart_suhu_data': json.dumps(suhu_data),
        'chart_kelembapan_data': json.dumps(kelembapan_data),
    }

    return render(request, 'prediksi/laporan.html', context)
