import datetime

# Lista de tiempos en formato HH:MM
time_strings = [
    "07:59", "05:54", "04:05", "03:10", "02:55", "05:20", "08:13", "04:14",
    "06:27", "07:20", "07:11", "05:10", "04:58", "03:44", "05:17", "05:49",
    "02:33", "03:15", "03:42", "03:33", "02:33", "06:02", "05:31", "06:11",
    "06:41", "06:06", "05:27", "02:05", "14:11", "06:32"
]

# Convertir a minutos totales
time_in_minutes = []
for time in time_strings:
    try:
        hours, minutes = map(int, time.split(":"))
        total_minutes = hours * 60 + minutes
        time_in_minutes.append(total_minutes)
    except ValueError:
        pass  # Ignorar valores mal formateados

minutes = []

# Calcular la media de los tiempos
mean = sum(time_in_minutes) / len(time_in_minutes)
print("Mean time:", datetime.timedelta(minutes=mean))