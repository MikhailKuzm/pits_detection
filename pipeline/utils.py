import cv2
import segmentation_models_pytorch as smp 
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import supervision as sv

def get_seg_model(encoder_name, checkpoint_path = f'segmantation/logs/Linknet/resnet34/version_0/checkpoints/epoch=199-step=9200.ckpt'): 
    seg_model = smp.create_model(
                'Linknet',
                encoder_name=encoder_name,
                in_channels=3,
                classes=1 )  
    new_weights = rename_state_dict(checkpoint_path, seg_model)
    seg_model.load_state_dict(new_weights) 
    seg_model.eval()
    return seg_model

def get_det_model(checkpoint_path = f'detection/logs/yolo11n/train2/weights/best.pt'):
    return YOLO(checkpoint_path)  

def compute_iou(box1, box2):
    """ Функция вычисления IoU (Intersection over Union) между двумя боксами """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Определяем координаты пересечения
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Вычисляем площадь пересечения
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Площади боксов
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)

    # Вычисляем IoU
    iou = intersection / float(area1 + area2 - intersection)
    return iou


def clean_segmentation_mask(pred_mask, min_region_size=250):
    """
    Очищает сегментационную маску от случайных пикселей, оставляя только главный регион.
    
    :param pred_mask: Входная бинарная маска (numpy array, 0 и 1).
    :param min_region_size: Минимальный размер области, которую считаем значимой.
    :return: Очищенная бинарная маска (0 и 1).
    """
    # Убедимся, что тип данных uint8 (требуется для OpenCV)
    mask = (pred_mask * 255).astype(np.uint8)

    # Морфологическое закрытие (убираем разрывы и одиночные пиксели)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Убираем мелкие шумные компоненты (поиск связных областей)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Если найдено несколько областей, оставляем только самую большую
    new_mask = np.zeros_like(mask)
    
    # Перебираем все найденные компоненты
    for i in range(1, num_labels):  # Первый индекс (0) — фон
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_region_size:  # Условие: оставить только большие регионы
            new_mask[labels == i] = 255  # Заполняем найденную область белым

    return (new_mask > 0).astype(np.uint8)  # Преобразуем обратно в 0 и 1


def get_mask(image_numpy, seg_model, checkpoint_path = f'segmantation/logs/Linknet/resnet34/version_0/checkpoints/epoch=199-step=9200.ckpt'):
    #checkpoint_path = f'logs/{model_name}/resnet34/version_0/checkpoints/epoch=199-step=9200.ckpt' 
    tensor_img = torch.tensor(image_numpy / 255, dtype=torch.float32)
    mean = torch.tensor([[[[0.4850]], [[0.4560]], [[0.4060]]]])
    std = torch.tensor([[[[0.2290]], [[0.2240]], [[0.2250]]]])
    with torch.no_grad():
        tensor_img = (tensor_img - mean) / std
        logits_mask = seg_model(tensor_img)
    prob_mask = logits_mask.sigmoid()
    pred_mask = (prob_mask > 0.1).float()
    
    return clean_segmentation_mask(pred_mask.squeeze().detach().numpy())
 
def draw_boxes(frame, filtered_boxes, filtered_confs, mode, img_h = 512, img_w = 512):
    """
    Отрисовывает боксы из filtered_boxes на изображении frame с вероятностями.
    
    :param frame: Исходное изображение (numpy array, BGR).
    :param filtered_boxes: Список отфильтрованных боксов [[x1, y1, x2, y2], ...].
    :param filtered_confs: Соответствующие вероятности боксов.
    :return: Обработанное изображение (PIL Image).
    """
    # Конвертируем BGR → RGB
    if mode == 'video':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(frame).resize((img_h, img_w))
    # Создаём объект для рисования
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Используем стандартный шрифт
    # Отрисовываем боксы и подписи вероятностей
    for box, conf in zip(filtered_boxes, filtered_confs):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Красные рамки
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="red", font=font)  # Надпись сверху
        
    return image  # Возвращаем PIL изображение


#frame = 'road-potholes.jpg'
def get_pred_mask(frame, total_detections_all, total_detections_seg,  mode,  
                  seg_model, det_model, conf, iou,  img_h = 512, img_w = 512 ):
    if mode == 'image':
        frame = cv2.imread(frame, 1) #загрузка изображэения по пути 
    img_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_original_resized = cv2.resize(img_original, (img_h,img_w))  
    img_original_transp = np.transpose(img_original_resized, (2, 0, 1)) 

    mask = get_mask(image_numpy = img_original_transp, seg_model = seg_model) #получение сегментированной маски дороги
    detect_result = det_model(frame, conf = conf, iou = iou)  
    # Извлекаем YOLO боксы (координаты в оригинальном размере изображения)
    yolo_boxes = detect_result[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
    yolo_confs = detect_result[0].boxes.conf.cpu().numpy()  # Вероятности предсказаний
    # Получаем оригинальные размеры изображения
    orig_h, orig_w, _ = frame.shape
    
    # Масштабируем YOLO боксы в размер mask (512x512)
    scale_x = img_w / orig_w
    scale_y = img_h / orig_h
    scaled_boxes = yolo_boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x1, x2
    scaled_boxes[:, [1, 3]] *= scale_y  # y1, y2
    scaled_boxes = scaled_boxes.astype(int)  # Округляем до целых пикселей
    
    # Фильтрация боксов (оставляем только те, у которых >=90% пикселей внутри mask == 1)
    filtered_boxes, filtered_confs = [], []
    for i, box in enumerate(scaled_boxes):
        x1, y1, x2, y2 = box
        # Избегаем выхода за границы (обрезаем координаты)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w-1, x2), min(img_h-1, y2)
        # Извлекаем область в mask, соответствующую текущему YOLO боксу
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:  # Если вдруг область пустая, пропускаем
            continue
        # Вычисляем, сколько пикселей попало в сегментированную зону
        inside_mask = np.sum(roi == 1)  # Количество пикселей внутри дороги
        box_area = (x2 - x1) * (y2 - y1)  # Площадь бокса

        if inside_mask / box_area >= 0.3:
            filtered_boxes.append(scaled_boxes[i])  # Добавляем бокс (в исходных размерах)
            filtered_confs.append(yolo_confs[i])  # Сохраняем вероятность
            
    # Преобразуем в numpy массивы
    filtered_boxes = np.array(filtered_boxes)
    filtered_confs = np.array(filtered_confs)
    
    if mode != 'image':
        global object_buffer
        global object_buffer_all
        OBJECT_BUFFER_SIZE = 8
        
        # Подсчет новых объектов, которых не было в предыдущем кадре и в object_buffer
        new_detections = 0
        new_detections_all = 0
        
        if len(object_buffer_all) == 0:
            # Если это первый кадр, учитываем все боксы
            new_detections = len(filtered_boxes)
            new_detections_all = len(scaled_boxes)
        else:  
            for box in filtered_boxes:
                # Проверяем, есть ли объект в последних кадрах  
                found_in_previous = any( any(compute_iou(box, prev_box) > 0.1 for prev_box in prev_frame) for prev_frame in object_buffer) 
                if not found_in_previous:  
                    new_detections += 1  # Объект новый

            for box in scaled_boxes:
                found_in_previous_all = any(any(compute_iou(box, prev_box) > 0.1 for prev_box in prev_frame) for prev_frame in object_buffer_all) 
                if not found_in_previous_all:  
                    new_detections_all += 1  # Объект новый 
                
                # Проверяем, есть ли объект в последних кадрах
        print('new_detections=', new_detections,  'new_detections_all = ', new_detections_all)
        total_detections_seg += new_detections  # Обновляем счетчик
        total_detections_all += new_detections_all 
        
        # Обновление object_buffer
        if len(object_buffer) >= OBJECT_BUFFER_SIZE:
            object_buffer.pop(0)  # Удаляем старые объекты
        if len(filtered_boxes) > 0:
            object_buffer.append(set(map(tuple, filtered_boxes)))  # Преобразуем в set, чтобы хранить уникальные боксы

        if len(object_buffer_all) >= OBJECT_BUFFER_SIZE:
            object_buffer_all.pop(0)  # Удаляем старые объекты
        if len(scaled_boxes) > 0:
            object_buffer_all.append(set(map(tuple, scaled_boxes))) 
    
    detected_img_segment = draw_boxes(img_original, filtered_boxes, filtered_confs, mode, img_h = 512, img_w = 512) 
    detected_img_all = draw_boxes(img_original, scaled_boxes, yolo_confs, mode, img_h = 512, img_w = 512)
    if mode != 'image':
        draw = ImageDraw.Draw(detected_img_segment)
        draw_all = ImageDraw.Draw(detected_img_all)
        # Загружаем шрифт (по умолчанию)
        font = font = ImageFont.load_default(30)  # Можно изменить размер шрифта 
        # Текст, который будем писать
        text = f"Count: {total_detections_seg}"
        text_all = f"Count: {total_detections_all}" 
        # Координаты (x, y)
        position = (20, 40) 
        # Цвет текста (зеленый)
        color = (0, 255, 0) 
        # Рисуем текст
        draw.text(position, text, fill=color, font=font)
        draw_all.text(position, text_all, fill=color, font=font)
            
        return detected_img_segment, detected_img_all, mask, total_detections_all, total_detections_seg
    return  detected_img_segment, detected_img_all, mask
 

def rename_state_dict(checkpoint_path, model):
    """
    Загружает checkpoint, переименовывает слои и возвращает новый state_dict.

    :param checkpoint_path: путь к сохранённому checkpoint'у
    :param model: текущая модель PyTorch, в которую загружаются веса
    :return: новый state_dict с правильными именами
    """
    # Загружаем checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Получаем текущие имена слоев в модели
    model_layers = set(model.state_dict().keys())

    # Создаём новый state_dict с исправленными названиями
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("model.", "")  # Убираем префикс 'model.', если он есть
        #new_key = new_key.replace("encoder.", "model.encoder.")  # Пример: исправляем 'encoder' → 'model.encoder'
        #new_key = new_key.replace("decoder.", "model.decoder.")  # Пример: исправляем 'decoder' → 'model.decoder'

        # Добавляем только те слои, которые есть в текущей модели
        if new_key in model_layers:
            new_state_dict[new_key] = value

    return new_state_dict

 
        
def save_det_results_as_video(results, output_video_path, fps=30):
    if not results:
        print("Ошибка: results пустой, нечего сохранять.")
        return
    # Получаем размеры кадра из первого результата
    first_frame = results[0].plot()  # Получаем numpy-массив кадра
    height, width, _ = first_frame.shape  # Извлекаем размеры
    
    # Кодек и создание VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for result in results:
        frame = result.plot()  # Получаем numpy-изображение с разметкой YOLO
        out.write(frame)  # Записываем кадр в видео

    # Освобождаем ресурсы
    out.release()
    print(f"Видео сохранено: {output_video_path}")
    

# Функция для обработки видео и сохранения двух выходных файлов
def process_video(input_video_path, output_video1_path, output_video2_path, output_video3_path,
                  seg_model, det_model, conf, iou, img_w=512, img_h=512):
    """
    Обрабатывает видео кадр за кадром, применяет perspective_transf, 
    и сохраняет два новых видео: transformed_road.mp4 и mask.mp4.

    :param input_video_path: Путь к входному видео
    :param output_video1_path: Путь для сохранения преобразованного видео 
    :param output_video2_path: Путь для сохранения преобразованного видео с учетом сегментации
    :param output_video3_path: Путь для сохранения видео маски
    :param model: Обученная модель сегментации
    :param img_w: Ширина выходного кадра
    :param img_h: Высота выходного кадра
    """ 
    # Открываем видеофайл
    cap = cv2.VideoCapture(input_video_path)
    
    # Проверяем, удалось ли открыть видео
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return

    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Кадры в секунду
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина входного видео
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота входного видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров

    print(f"Обрабатываем видео: {input_video_path}")
    print(f"Разрешение: {frame_width}x{frame_height}, FPS: {fps}, Всего кадров: {total_frames}")

    # Кодек и создание VideoWriter для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    out1 = cv2.VideoWriter(output_video1_path, fourcc, fps, (img_w, img_h))  # Преобразованное видео без наложения сегментации
    out2 = cv2.VideoWriter(output_video2_path, fourcc, fps, (img_w, img_h))  # Преобразованное видео 
    out3 = cv2.VideoWriter(output_video3_path, fourcc, fps, (img_w, img_h), isColor=False)  # Маска (ч/б видео)

    frame_count = 0  # Счётчик кадров
    total_detections_all = 0
    total_detections_seg = 0
    #previous_bboxes, previous_bboxes_all = [], []
    total_detections_all, total_detections_seg = 0, 0 
    # Читаем видео кадр за кадром
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Если видео закончилось, выходим из цикла 

        # Обрабатываем кадр через perspective_transf
        detected_img_segment, detected_img_all, mask,  total_detections_all,\
        total_detections_seg = get_pred_mask(frame,  total_detections_all, total_detections_seg,
                                                                     mode = 'video', seg_model = seg_model, det_model = det_model,
                                                                     conf = conf, iou = iou, img_h = img_h, img_w = img_w)
        print(total_detections_seg, total_detections_all)
        # Преобразуем изображения в правильный формат для сохранения
       # transformed_frame_bgr = cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR)  # Обратно в BGR для OpenCV
        mask_uint8 = (mask * 255).astype(np.uint8)  # Маска должна быть 0 или 255

        # Записываем кадры в выходные видеофайлы
        out1.write(np.array(detected_img_all))  # Преобразованное видео
        out2.write(np.array(detected_img_segment))
        out3.write(mask_uint8)  # Бинарная маска

        frame_count += 1
        if frame_count % 50 == 0:  # Выводим статус каждые 50 кадров
            print(f"Обработано {frame_count}/{total_frames} кадров...")

    # Освобождаем ресурсы
    cap.release()
    out1.release()
    out2.release()
    out3.release()
    cv2.destroyAllWindows()

    print(f"Обработка завершена! Видео сохранены в:\n 1) {output_video1_path}\n 2) {output_video2_path} \n 3) {output_video3_path}")
    
    
def merge_videos(video_paths, output_path, resize_height=None):
    """
    Объединяет несколько видео в одно, располагая кадры горизонтально.

    :param video_paths: Список путей к видеофайлам в нужном порядке.
    :param output_path: Путь для сохранения объединенного видео.
    :param resize_height: Высота видео на выходе (автоматически подгоняет ширину).
    """
    
    # Открываем видеофайлы
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    
    # Проверяем, удалось ли открыть все видео
    if not all([cap.isOpened() for cap in caps]):
        print("Ошибка: Не удалось открыть одно или несколько видеофайлов.")
        return

    # Получаем параметры первого видео (будем использовать их за основу)
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    frame_widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
    frame_heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
    min_frames = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])  # Чтобы видео не заканчивалось раньше других
    
    print(f"Обнаружено {len(video_paths)} видео. Используем FPS={fps}, минимальная длина={min_frames} кадров")

    # Устанавливаем высоту кадров (если не задано, используем минимальную)
    if resize_height is None:
        resize_height = min(frame_heights)

    # Определяем общий размер выходного видео
    total_width = sum([int(w * (resize_height / h)) for w, h in zip(frame_widths, frame_heights)])  # Подгон ширины

    # Создаём VideoWriter для сохранения объединенного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, resize_height))

    frame_count = 0

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break  # Останавливаемся, если одно из видео закончилось

            # Изменяем размер кадра, чтобы высота соответствовала `resize_height`
            new_width = int(frame.shape[1] * (resize_height / frame.shape[0]))  # Масштабируем ширину
            resized_frame = cv2.resize(frame, (new_width, resize_height))
            frames.append(resized_frame)

        if len(frames) != len(video_paths):
            break  # Если не удалось считать все кадры, выходим

        # Объединяем кадры в один кадр по горизонтали
        merged_frame = cv2.hconcat(frames)

        # Записываем объединенный кадр в выходное видео
        out.write(merged_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Обработано {frame_count}/{min_frames} кадров...")

    # Освобождаем ресурсы
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Объединённое видео сохранено: {output_path}")