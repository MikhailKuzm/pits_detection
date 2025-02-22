import os
import shutil

def move_files(src_folder, masks_folder, images_folder):
    # Получаем список всех файлов в исходной папке
    files = os.listdir(src_folder)
    
    # Проходим по каждому файлу
    for file in files:
        # Получаем полный путь к файлу
        file_path = os.path.join(src_folder, file)
        
        # Проверяем, что это файл (а не директория)
        if os.path.isfile(file_path):
            if file.endswith('_mask.jpg') or file.endswith('_mask.png'):  # Можно добавить другие расширения
                # Если файл - маска, перемещаем его в папку masks_folder
                shutil.move(file_path, os.path.join(masks_folder, file))
            else:
                # Если файл - изображение, перемещаем его в папку images_folder
                shutil.move(file_path, os.path.join(images_folder, file))

# Папки для train и valid
train_folder = 'train'
valid_folder = 'valid'

# Папки назначения
masks_train_folder = 'seg_dataset/masks_train'
images_train_folder = 'seg_dataset/images_train'
masks_valid_folder = 'seg_dataset/masks_val'
images_valid_folder = 'seg_dataset/images_val'

# Создаем папки назначения, если их нет
os.makedirs(masks_train_folder, exist_ok=True)
os.makedirs(images_train_folder, exist_ok=True)
os.makedirs(masks_valid_folder, exist_ok=True)
os.makedirs(images_valid_folder, exist_ok=True)

# Перемещаем файлы из папки train
move_files(train_folder, masks_train_folder, images_train_folder)

# Перемещаем файлы из папки valid
move_files(valid_folder, masks_valid_folder, images_valid_folder)
