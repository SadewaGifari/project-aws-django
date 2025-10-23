# akun/views.py

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # Simpan pengguna baru ke database
            login(request, user)  # Langsung login setelah registrasi berhasil
            return redirect('dashboard')  # Arahkan ke halaman utama/dashboard
    else:
        form = UserCreationForm()
    
    return render(request, 'akun/register.html', {'form': form})