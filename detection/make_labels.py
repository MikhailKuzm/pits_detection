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