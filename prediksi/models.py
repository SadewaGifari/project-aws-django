from django.db import models

# Create your models here.
class DataSensor(models.Model):
    temperature = models.FloatField()
    humidity = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Suhu: {self.temperature}°C, Lembap: {self.humidity}% pada {self.timestamp}"