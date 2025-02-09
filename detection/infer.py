import os
import torch
import cv2
import hydra
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from detection.detection_model import ObjectDetectionModel


def load_best_checkpoint(model_name: str):
    """Находит и загружает лучший чекпоинт модели"""
    checkpoint_dir = f"logs/{model_name}/checkpoints"
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
        key=lambda x: float(x.split("val_mAP-")[-1].replace(".ckpt", "")),  # Сортируем по лучшему mAP
        reverse=True
    )
    if not checkpoints:
        raise FileNotFoundError(f"Нет сохранённых чекпоинтов в {checkpoint_dir}")
    
    best_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    print(f"Загружаем лучший чекпоинт: {best_checkpoint}")
    return best_checkpoint


def preprocess_image(image_path):
    """Загружает изображение, переводит в формат, необходимый сети"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0), image  # Возвращаем тензор и исходное изображение


def draw_predictions(image, boxes, scores, labels, score_threshold=0.5):
    """Рисует bounding boxes на изображении"""
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < score_threshold:
            continue

        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


@hydra.main(version_base=None, config_path="configs", config_name="infer")
def infer(cfg: DictConfig):
    """Основная функция для инференса"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем лучший чекпоинт
    best_checkpoint = load_best_checkpoint(cfg.model.name)

    # Загружаем модель
    model = ObjectDetectionModel.load_from_checkpoint(best_checkpoint, model_type=cfg.model.name,
                                                      num_classes=cfg.model.num_classes, lr=cfg.model.lr)
    model.to(device)
    model.eval()

    # Создаём папку для результатов
    os.makedirs("detection/results", exist_ok=True)

    # Предсказания для каждого изображения
    for image_name in os.listdir(cfg.infer.input_dir):
        if not image_name.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(cfg.infer.input_dir, image_name)
        image_tensor, orig_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)

        # Предсказание
        with torch.no_grad():
            predictions = model(image_tensor)[0]

        # Обработка результатов
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        # Отрисовка боксов
        output_image = draw_predictions(orig_image, boxes, scores, labels)

        # Сохранение результата
        output_path = os.path.join("detection/results", f"pred_{image_name}")
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    infer()