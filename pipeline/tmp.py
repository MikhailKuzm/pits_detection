from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp 
from ultralytics import YOLO
import os
from pipeline.utils import *

seg_model = get_seg_model('resnet101','segmantation/logs/Linknet/resnet101/version_4/checkpoints/epoch=79-step=1360.ckpt')
det_model = get_det_model('detection/logs/yolo11n/train2/weights/best.pt')
def main(mode, frame_path, save_dir, merge_image = True, conf = 0.6, iou = 0.1, img_h = 512, img_w = 512): 
    if mode == 'video':
         
        process_video(input_video_path=frame_path,#frame_path, 
                    output_video1_path=f"{save_dir}/detected_road.mp4",
                    output_video2_path=f"{save_dir}/detected_road_segment.mp4",  
                    output_video3_path=f"{save_dir}/mask.mp4", 
                    seg_model=seg_model,det_model = det_model, conf = conf, iou =iou,  img_w=img_w, img_h=img_h)
        # Параметры видеофайлов
        video_files = [frame_path, f"{save_dir}/mask.mp4", 
                       f"{save_dir}/detected_road_segment.mp4", f"{save_dir}/detected_road.mp4"]
        output_video = "pipeline/results/merged_video.mp4"
        # Запускаем объединение видео
        merge_videos(video_files, output_video) 
    elif mode == 'image':
        detected_img_segment, detected_img_all, mask = get_pred_mask(frame = frame_path, mode = mode, seg_model = seg_model, 
                                                                     det_model = det_model, img_w = img_w, img_h = img_h, conf = conf, iou = iou)
        if merge_image:
            # Загружаем первое изображение (BGR → RGB, масштабируем до 512x512)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame_img = Image.fromarray(frame) 
            # Преобразуем бинарную маску (numpy → PIL, черно-белое изображение)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")  # L = Grayscale 
            # Создаём пустое изображение шириной 1536 (512 * 3)
            merged_width = 512 * 4
            merged_img = Image.new("RGB", (merged_width, 512)) 
            # Вставляем изображения в итоговое изображение
            merged_img.paste(frame_img, (0, 0))           # Оригинальное изображение
            merged_img.paste(mask_img.convert("RGB"), (512, 0))  # Маска (перевод в RGB)
            merged_img.paste(detected_img_segment, (1024, 0))     # YOLO детекции
            merged_img.paste(detected_img_all, (1536, 0))     # YOLO детекции
            merged_img.save(f'{save_dir}/merged_output_{frame_path[-8:-4]}.jpg')
        else:
            img = Image.fromarray(mask*255)
            img.save(f'{save_dir}/mask_{frame_path[-8:-4]}.jpg')
            detected_img_segment.save(f'{save_dir}/detect_seg_{frame_path[-8:-4]}.jpg')
            detected_img_all.save(f'{save_dir}/detect_all_{frame_path[-8:-4]}.jpg')
        

if __name__ == 'main':
    object_buffer, object_buffer_all = [], []
    main(mode = 'video', frame_path = 'sample_video.mp4', save_dir = 'pipeline/results/', 
         merge_image = True, img_h = 512, img_w = 512) 