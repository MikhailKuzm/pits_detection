from flask import Flask, render_template, request, jsonify,  send_from_directory
import cv2 
from flask_socketio import SocketIO, emit
import random
import os
from werkzeug.utils import secure_filename
import numpy as np
from ultralytics import YOLO 
import torch
import segmentation_models_pytorch as smp 
import subprocess


app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")  


UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/results/<filename>')
def send_video(filename): 
    video_folder = 'static/results'
    video_path = os.path.join(video_folder, filename)
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    print(f"Video path: {video_path}")
    print('size = ', os.path.getsize(video_path))
    
    
    if not os.path.exists(video_path):
        print('Video not found')
        return "Video not found", 404
    return send_from_directory(video_folder, filename,  mimetype='video/mp4') 



@app.route('/upload_video', methods=['POST'])
def upload_video():
    #print("Inside upload_video")
    
    if 'file' in request.files:
        file = request.files['file']
        #print(f"Got File: {file.filename}")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result_video_name = process_video(file_path)
            #os.remove(file_path)
            print(f'Send to client side video: {result_video_name}')
            return jsonify({'message': 'Видео успешно загружено', 'video_path': result_video_name}), 200
        else:
            return jsonify({'error': 'Нет файла для загрузки'}), 400

    return jsonify({'error': 'Неверный формат запроса'}), 400

def compute_iou(box1, box2):
    """ Функция вычисления IoU (Intersection over Union) между двумя боксами """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2 
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g) 
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height 
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)

    
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def clean_segmentation_mask(pred_mask, min_region_size=512): 
    mask = (pred_mask * 255).astype(np.uint8) 
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) 
    new_mask = np.zeros_like(mask) 
    for i in range(1, num_labels):  
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_region_size:
            new_mask[labels == i] = 255  
    return (new_mask > 0).astype(np.uint8)

def get_seg_model(checkpoint_path = 'static/models_weights/best_seg.ckpt'): 
    seg_model = smp.create_model(
                'Linknet',
                encoder_name='resnet101',
                in_channels=3,
                classes=1)  
    seg_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    seg_model.eval()
    return seg_model

def get_mask(image_numpy, seg_model):
    tensor_img = torch.tensor(image_numpy / 255, dtype=torch.float32)
    mean = torch.tensor([[[[0.4850]], [[0.4560]], [[0.4060]]]])
    std = torch.tensor([[[[0.2290]], [[0.2240]], [[0.2250]]]])
    with torch.no_grad():
        tensor_img = (tensor_img - mean) / std
        logits_mask = seg_model(tensor_img)
    prob_mask = logits_mask.sigmoid()
    pred_mask = (prob_mask > 0.2).float()
    
    return clean_segmentation_mask(pred_mask.squeeze().detach().numpy())
 

def process_video(video_path):
    #print('Inside process_video')
    if not os.path.exists(video_path):
        print(f"Video file not found in process_video function: {video_path}")
        return None
    cap = cv2.VideoCapture(video_path) 

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    
    line_y = int(3 * height / 4)  
    
    intermediate_video_name = f'results/{random.randint(0, 64)}_intermediate.mp4.mp4' 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter('static/' + intermediate_video_name, fourcc, fps, (width*2, height))

    det_model = YOLO('static/models_weights/best.pt')
    seg_model = get_seg_model('static/models_weights/best_seg.ckpt')
    current_frame = 0  
    img_h,img_w = 224, 224
    object_buffer = []
    OBJECT_BUFFER_SIZE = 5
    
    new_detections = 0 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_h, orig_w, _ = frame.shape
        img_original_resized = cv2.resize(frame, (img_h,img_w))  
        img_original_transp = np.transpose(img_original_resized, (2, 0, 1)) 
        mask = get_mask(image_numpy = img_original_transp, seg_model = seg_model) 
        
        detect_result = det_model(frame, conf=0.5, iou=0.4, verbose=False)
        yolo_boxes = detect_result[0].boxes.xyxy.cpu().numpy()
        yolo_confs = detect_result[0].boxes.conf.cpu().numpy()
        
        
        scale_x = img_w / orig_w
        scale_y = img_h / orig_h
        scaled_boxes = yolo_boxes.copy()
        scaled_boxes[:, [0, 2]] *= scale_x 
        scaled_boxes[:, [1, 3]] *= scale_y 
        scaled_boxes = scaled_boxes.astype(int) 
        
        filtered_box_ind = []
        for i, box in enumerate(scaled_boxes):
            x1, y1, x2, y2 = box
            
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w-1, x2), min(img_h-1, y2)
            
            roi = mask[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            inside_mask = np.sum(roi == 1) 
            box_area = (x2 - x1) * (y2 - y1)  

            if inside_mask / box_area >= 0.3:
                filtered_box_ind.append(i)

        for box, conf in zip(yolo_boxes[filtered_box_ind], yolo_confs[filtered_box_ind]):
            x1, y1, x2, y2 = map(int, box)    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   
            if len(object_buffer): 
                if not any(any(compute_iou(box, prev_box) > 0.1 for prev_box in prev_frame) for prev_frame in object_buffer):
                    print(new_detections)
                    new_detections += 1   
                
        
        if len(object_buffer) >= OBJECT_BUFFER_SIZE:
            object_buffer.pop(0)  
        if len(filtered_box_ind) > 0:
            object_buffer.append(set(map(tuple, yolo_boxes[filtered_box_ind]))) 
        cv2.putText(frame, f"Found: {new_detections}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  

        mask_resized = cv2.resize(mask*255, (width, height))   
        mask_resized = np.stack([mask_resized] * 3, axis=-1)
       
        combined_frame = np.hstack((mask_resized, frame))  
        out.write(combined_frame)
        progress = int((current_frame / total_frames) * 100)
        socketio.emit('progress', {'progress': progress})
        print('progress', progress) 

        current_frame += 1  
        
    cap.release()
    out.release()
    socketio.emit('progress', {'progress': 100})
    
     
    result_video_name = f'results/{random.randint(0, 64)}.mp4'
    ffmpeg_command = [
        'ffmpeg', '-i', f'static/{intermediate_video_name}',
        '-vcodec', 'libx264', f'static/{result_video_name}'
    ]
    subprocess.call(ffmpeg_command)
 
    os.remove(f'static/{intermediate_video_name}') 

    return result_video_name 


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
