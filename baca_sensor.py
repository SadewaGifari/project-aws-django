import serial
import requests
import time
import json

SERIAL_PORT = 'COM6'
BAUD_RATE = 9600
API_URL = 'http://127.0.0.1:8000/api/simpan-data-sensor/'

print("Mencoba terhubung ke port serial...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"Berhasil terhubung ke {SERIAL_PORT}")
    time.sleep(2)  # beri waktu Arduino untuk reset

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if not line:
            print("Menunggu data dari Arduino...")
            continue

        print(f"Menerima data mentah: {line}")

        if "Suhu" in line and "Kelembapan" in line:
            try:
                parts = line.split('\t')
                humidity_str = parts[0].split(':')[1].replace('%', '').strip()
                temperature_str = parts[1].split(':')[1].replace('*C', '').strip()

                humidity = float(humidity_str)
                temperature = float(temperature_str)

                payload = {'temperature': temperature, 'humidity': humidity}
                print(f"Data yang akan dikirim: {payload}")

                response = requests.post(API_URL, json=payload)
                if response.status_code == 201:
                    print(">> Sukses: Data berhasil dikirim ke server Django.")
                else:
                    print(f">> Gagal: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"!! Error parsing: {e}")

        time.sleep(1)

except serial.SerialException as e:
    print(f"!! Gagal terhubung ke {SERIAL_PORT}: {e}")
