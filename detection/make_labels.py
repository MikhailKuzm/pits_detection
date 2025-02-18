import os
import xml.etree.ElementTree as ET

# Директория с XML и изображениями
dataset_path = "detection\dataset\potholes"

# Класс для YOLO (если только один класс "pothole", то 0)
CLASS_ID = 0

# Проход по файлам в папке
for filename in os.listdir(dataset_path):
    if filename.endswith(".xml"):
        xml_path = os.path.join(dataset_path, filename)
        txt_path = os.path.join(dataset_path, filename.replace(".xml", ".txt"))

        # Разбираем XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Получаем размер изображения
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        # Открываем txt-файл для записи
        with open(txt_path, "w") as txt_file:
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # YOLO формат: class x_center y_center width height (все отнормированные)
                x_center = (xmin + xmax) / (2.0 * img_width)
                y_center = (ymin + ymax) / (2.0 * img_height)
                box_width = (xmax - xmin) / img_width
                box_height = (ymax - ymin) / img_height

                # Записываем в txt файл
                txt_file.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        print(f"Создан файл: {txt_path}")





import os
import shutil
import random

# Пути к папкам
source_dir = "detection/data/potholes"  # Исходная папка с изображениями и метками
train_img_dir = "detection/data/train/images"
train_lbl_dir = "detection/data/train/labels"
valid_img_dir = "detection/data/valid/images"
valid_lbl_dir = "detection/data/valid/labels"

# Создаём папки, если их нет
for path in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
    os.makedirs(path, exist_ok=True)

# Получаем список файлов
files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]
random.shuffle(files)  # Перемешиваем файлы для случайного разбиения

# Определяем границу разбиения
split_idx = int(len(files) * 0.8)

# Функция копирования файлов
def copy_files(file_list, dest_img_dir, dest_lbl_dir):
    for file in file_list:
        img_path = os.path.join(source_dir, file)
        txt_path = os.path.join(source_dir, file.replace(".jpg", ".txt"))

        # Копируем, если есть и изображение, и метки
        if os.path.exists(txt_path):
            shutil.copy(img_path, os.path.join(dest_img_dir, file))
            shutil.copy(txt_path, os.path.join(dest_lbl_dir, file.replace(".jpg", ".txt")))

# Копируем файлы в `train`
copy_files(files[:split_idx], train_img_dir, train_lbl_dir)

# Копируем файлы в `valid`
copy_files(files[split_idx:], valid_img_dir, valid_lbl_dir)

print(f"✅ Данные разделены: {len(files[:split_idx])} train, {len(files[split_idx:])} valid")




import os

def replace_class_in_labels(label_dir):
    """
    Проходит по всем .txt файлам в директории label_dir и заменяет класс 0 на 1.
    """
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):  # Обрабатываем только .txt файлы
            file_path = os.path.join(label_dir, filename)

            # Читаем файл
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Заменяем класс 0 на 1
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == "1":  # Если класс 0
                    parts[0] = "0"  # Меняем на 1
                updated_lines.append(" ".join(parts))  # Объединяем обратно

            # Записываем обратно в файл
            with open(file_path, "w") as f:
                f.write("\n".join(updated_lines))

            print(f"✅ Обновлён файл: {file_path}")

# 📌 Заменяем классы в папках train/labels и valid/labels
replace_class_in_labels("detection/data/yolo_format/labels/train")
replace_class_in_labels("detection/data/yolo_format/labels/val")

print("🎯 Все классы 0 заменены на 1!")



import os

# Пути к папкам
image_dir = "detection/data/train/images"
label_dir = "detection/data/train/labels"

# Получаем список изображений без расширений
image_names = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))}

# Проверяем файлы в папке labels
for label_file in os.listdir(label_dir):
    label_name, ext = os.path.splitext(label_file)

    # Если файл .txt, но нет соответствующего изображения — удаляем
    if ext == ".txt" and label_name not in image_names:
        os.remove(os.path.join(label_dir, label_file))
        print(f"Удалён: {label_file}")

print("Очистка завершена.")