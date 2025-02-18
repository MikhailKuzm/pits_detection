import os
import torch
from PIL import Image, ImageDraw, ImageFont
import hydra
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from detection.detection_model import ObjectDetectionModel
from torchvision import transforms
from torchvision.ops import nms
from ultralytics import YOLO
os.chdir('detection')

def load_best_checkpoint(model_name: str):
    """Находит и загружает лучший чекпоинт модели"""
    checkpoint_dir = f"logs/{model_name}/checkpoints"
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=lambda x: float(x.split("val_mAP=")[-1].replace(".ckpt", "")),  # Сортируем по лучшему mAP
        reverse=True
    )
    if not checkpoints:
        raise FileNotFoundError(f"Нет сохранённых чекпоинтов в {checkpoint_dir}")
    
    best_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    print(f"Загружаем лучший чекпоинт: {best_checkpoint}")
    return best_checkpoint


def preprocess_image(image_path):
    """Загружает изображение, переводит в формат, необходимый сети"""
    to_tensor = transforms.ToTensor()
    image =  Image.open(image_path).convert('RGB').resize((576,576)) 
    tensor =  to_tensor(image)
    return tensor.unsqueeze(0), image  # Возвращаем тензор и исходное изображение


def draw_predictions(image, boxes, scores, labels, score_threshold=0.5):
    """Рисует bounding boxes на изображении"""
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    img1 = ImageDraw.Draw(image)    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < score_threshold:
            continue

        x_min, y_min, x_max, y_max = map(int, box)
        img1.rectangle((x_min, y_min, x_max, y_max), outline = (0, 0, 255), width= 3)
        img1.text((x_min, y_min - 10), f"{score:.2f}", font=fnt )
    return image


import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="infer", version_base=None)
def load_config(cfg):
    # 🔄 Конвертируем Hydra-объект в обычный Python-словарь
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # ✅ Выводим загруженный конфиг
    print(config_dict)
    
    return config_dict

if __name__ == "__main__":
    cfg = load_config()



@hydra.main(version_base=None, config_path="configs", config_name="infer")
def infer(cfg: DictConfig):
    """Основная функция для инференса"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.model.name == 'yolo':
        for model_weight in ["yolo11x"]:
            os.makedirs(f"results/{model_weight}", exist_ok=True)
            model = YOLO(f"logs/{model_weight}/train/weights/best.pt")
            # Run batched inference on a list of images
            results = model([f'data/valid/images/{img_name}' for img_name in os.listdir('data/valid/images')], conf = 0.2, iou = 0.2)
            # Process results list
            for result in results:
                name = result.path.split('/')[-1]
                boxes = result.boxes  # Boxes object for bounding box outputs
                result.save(filename=f"results/{model_weight}/{name}")  # save to disk 
        return
    # Загружаем лучший чекпоинт
    best_checkpoint = load_best_checkpoint(cfg.model.name) #

    # Загружаем модель
    model = ObjectDetectionModel.load_from_checkpoint(best_checkpoint)
    model.to(device)
    model.eval()

    # Создаём папку для результатов
    os.makedirs(f"results/{cfg.model.name}", exist_ok=True)

    # Предсказания для каждого изображения
    for image_name in os.listdir('data/valid/images'):#cfg.infer.input_dir
        if not image_name.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join('data/valid/images', image_name)
        image_tensor, orig_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)

        # Предсказание
        with torch.no_grad():
            predictions = model(image_tensor)[0]

        # Обработка результатов
        boxes = predictions["boxes"].cpu()#.numpy()
        scores = predictions["scores"].cpu()#.numpy()
        labels = predictions["labels"].cpu()#.numpy()
        indx = nms(boxes, scores, iou_threshold=0.2)
        boxes = boxes[indx]
        scores = scores[indx]
        labels = labels[indx] 

        # Отрисовка боксов
        output_image = draw_predictions(image = orig_image, boxes = boxes, scores=scores, labels=labels)

        # Сохранение результата
        output_path = os.path.join(f"results/{cfg.model.name}", f"pred_{image_name}")
        output_image.save(output_path)
        print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    infer()
    
    
    
    
    
from ultralytics import YOLO
model = YOLO(f"detection/logs/yolo11n/train/weights/best.pt")
results = model('detection/test_images/sample_video.mp4', save = True, conf = 0.1, iou = 0.2)
for result in results:
    name = result.path.split('/')[-1]
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.save(filename=f"pred{name}")  # save to disk 

