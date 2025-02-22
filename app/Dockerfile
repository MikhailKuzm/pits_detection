# Используем официальный образ Python
FROM python:3.12

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем локальные файлы приложения в контейнер
COPY . /app
RUN chmod -R a+rwx /app/uploads /app/static/results 
RUN apt-get update

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

#RUN apt-get update && apt-get install -y \ wget \ build-essential \ pkg-config \ cmake \ ninja-build \ libgl1-mesa-glx \ libleptonica-dev \ ffmpeg \ zlib1g-dev \ libncurses5-dev \ libgdbm-dev \ libnss3-dev \ libssl-dev \ libreadline-dev \ libffi-dev \ python3.8 \ python3-pip \ python3-opencv
RUN apt-get install python3-opencv -y

# Устанавливаем зависимости
# Обновляем pip до последней версии и устанавливаем зависимости
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
 
# Открываем порты, которые будут использоваться контейнером
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "app.py"]