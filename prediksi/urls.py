# prediksi/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_prediksi, name='dashboard'),
    path('laporan/', views.laporan_historis, name='laporan_historis'),
    path('api/simpan-data-sensor/', views.simpan_data_sensor, name='simpan_data_sensor'),
]