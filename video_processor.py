import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from PIL import Image
import logging
import psutil
import time
from convert import get_embedding
from kalmanfilter import KalmanFilter

# Setup logging
logging.basicConfig(level='INFO')  # Will be overridden by args.log_level in process_video

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_video(args, model, device):
    """Process video with human tracking, face embedding, DB matching, and re-identification."""
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if not args.video_path or not os.path.exists(args.video_path):
        raise ValueError(f"Video not found: {args.video_path}")
    
    # Load DB
    with open(args.db_path, 'r') as f:
        db = json.load(f)
    db_embeddings = {name: np.array(emb) for name, emb in db.items() if emb}
    if not db_embeddings:
        raise ValueError("Empty DB - build it first with --build_db")
    
    # Load YOLOv8
    yolo = YOLO('yolov8m.pt') 
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Re-ID structures
    known_persons = {}  # person_id -> {'avg_emb': np.array or None, 'name': str or None, 'last_score': float or None}
    next_person_id = 0
    
    # Track state: track_id -> {'embeddings': deque, 'person_id': int or None, 'frame_count': 0, 'kalman': KalmanFilter, 'last_detection': np.array, 'missed_frames': 0}
    tracks = defaultdict(lambda: {'embeddings': deque(maxlen=args.max_embs_per_track), 'person_id': None, 'frame_count': 0, 'kalman': None, 'last_detection': None, 'missed_frames': 0})
    
    frame_num = 0
    frame_times = deque(maxlen=10)  # For rolling FPS
    start_time = time.time()
    
    while cap.isOpened():
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        orig_frame = frame.copy()  # For annotation
        
        # YOLO track persons (every frame)
        results = yolo.track(frame, persist=True, classes=0, conf=args.conf_threshold, iou=args.iou_threshold)
        
        detected_tracks = set()  # Tracks detected this frame
        
        for result in results[0].boxes:
            if not result.id:
                continue
            track_id = int(result.id)
            bbox = result.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            detected_tracks.add(track_id)
            
            # Initialize Kalman if new
            if tracks[track_id]['kalman'] is None:
                tracks[track_id]['kalman'] = KalmanFilter()
                measurement = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], np.float32)
                tracks[track_id]['kalman'].kf.statePost = np.concatenate((measurement, [0, 0, 0, 0])).astype(np.float32)
            
            # Update Kalman with detection
            measurement = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], np.float32)
            tracks[track_id]['kalman'].update(measurement)
            tracks[track_id]['last_detection'] = bbox
            tracks[track_id]['missed_frames'] = 0
            
            tracks[track_id]['frame_count'] += 1
            
            # Attempt embedding every skip_interval frames
            if tracks[track_id]['frame_count'] % args.skip_interval == 0:
                # Crop upper body/head with padding
                person_h = bbox[3] - bbox[1]
                person_w = bbox[2] - bbox[0]
                pad_h = person_h * args.padding_ratio
                pad_w = person_w * args.padding_ratio
                crop_y1 = max(0, int(bbox[1] - pad_h))
                crop_y2 = min(height, int(bbox[1] + person_h * args.upper_crop_ratio + pad_h))
                crop_x1 = max(0, int(bbox[0] - pad_w))
                crop_x2 = min(width, int(bbox[2] + pad_w))
                
                if (crop_y2 - crop_y1 < args.min_crop_size) or (crop_x2 - crop_x1 < args.min_crop_size):
                    logging.debug(f"Frame {frame_num}, Track {track_id}: Crop too small, skipping.")
                    continue
                
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]  # BGR np array
                
                # InsightFace embedding (detection + alignment + normed emb)
                emb = get_embedding(bgr_np_image=crop, model_name=args.model_name, det_thresh=args.det_thresh)
                if emb is None:
                    logging.debug(f"Frame {frame_num}, Track {track_id}: No face detected in crop.")
                    continue
                
                tracks[track_id]['embeddings'].append(emb)
                logging.debug(f"Frame {frame_num}, Track {track_id}: Added embedding (total {len(tracks[track_id]['embeddings'])})")
            
            # Match/Re-ID if enough embeddings
            if len(tracks[track_id]['embeddings']) >= args.min_embs_for_match:
                avg_emb = np.mean(tracks[track_id]['embeddings'], axis=0)
                
                # Find best DB match
                max_db_sim = -1
                best_name = None
                for name, db_emb in db_embeddings.items():
                    sim = cosine_similarity(avg_emb, db_emb)
                    if sim > max_db_sim:
                        max_db_sim = sim
                        best_name = name
                name = best_name if max_db_sim > args.cos_threshold else None
                if name:
                    logging.info(f"Frame {frame_num}, Track {track_id}: DB match to {name} (sim={max_db_sim:.2f})")
                
                # Re-ID: Check if matches a known person
                matched_pid = None
                max_reid_sim = -1
                for pid, data in known_persons.items():
                    if data['avg_emb'] is not None:
                        sim = cosine_similarity(avg_emb, data['avg_emb'])
                        if sim > args.reid_threshold and sim > max_reid_sim:
                            if (data['name'] == name) or (data['name'] is None and name is None):
                                matched_pid = pid
                                max_reid_sim = sim
                                break
                            elif sim > max_reid_sim:
                                matched_pid = pid
                                max_reid_sim = sim
                
                if matched_pid is not None:
                    # Merge to existing person
                    tracks[track_id]['person_id'] = matched_pid
                    old_emb = known_persons[matched_pid]['avg_emb']
                    known_persons[matched_pid]['avg_emb'] = (old_emb + avg_emb) / 2
                    if name and known_persons[matched_pid]['name'] is None:
                        known_persons[matched_pid]['name'] = name
                    known_persons[matched_pid]['last_score'] = max_db_sim if name else None
                    logging.info(f"Frame {frame_num}, Track {track_id}: Re-ID matched to Person {matched_pid} (sim={max_reid_sim:.2f})")
                else:
                    # New person
                    person_id = next_person_id
                    next_person_id += 1
                    known_persons[person_id] = {'avg_emb': avg_emb, 'name': name, 'last_score': max_db_sim if name else None}
                    tracks[track_id]['person_id'] = person_id
                    logging.info(f"Frame {frame_num}, Track {track_id}: New Person {person_id}")
        
        # Handle missed detections with Kalman prediction
        for track_id, state in list(tracks.items()):
            if track_id not in detected_tracks:
                if state['kalman'] is not None and state['missed_frames'] < args.max_missed_frames:
                    predicted = state['kalman'].predict()
                    pred_x1, pred_y1, pred_w, pred_h = predicted[:4]
                    pred_bbox = np.array([pred_x1, pred_y1, pred_x1 + pred_w, pred_y1 + pred_h])
                    state['last_detection'] = pred_bbox
                    state['missed_frames'] += 1
                else:
                    # Remove old tracks
                    del tracks[track_id]
                    continue
            
            # Annotate using last_detection (detected or predicted)
            bbox = state['last_detection']
            if bbox is not None:
                pid = state['person_id']
                if pid is None:
                    label = "No face detected"
                    score = None
                    display_id = track_id
                else:
                    person_data = known_persons[pid]
                    label = person_data['name'] if person_data['name'] else "Unknown"
                    score = person_data['last_score']
                    display_id = pid
                
                # Color-coding
                if label != "Unknown" and label != "No face detected":  # Matched
                    color = (0, 255, 0)  # Green
                elif label == "Unknown":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Append score if matched
                display_label = f"Person {display_id}: {label}"
                if score is not None:
                    display_label += f" ({score:.2f})"
                
                # Draw bbox and text
                cv2.rectangle(orig_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(orig_frame, display_label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Verbose: Print bbox
                if args.verbose:
                    logging.debug(f"Frame {frame_num}, Track {track_id}: BBox {bbox}")
        
        # Timing: FPS in top-right
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        avg_fps = 1 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        cv2.putText(orig_frame, f"FPS: {avg_fps:.1f}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resources: GPU mem (if CUDA) and CPU % in bottom-left
        if 'cuda' in args.device:
            gpu_used = torch.cuda.memory_allocated() / 1024**2
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_str = f"GPU: {gpu_used:.0f}/{gpu_total:.0f} MB"
        else:
            gpu_str = "GPU: N/A"
        cpu_percent = psutil.cpu_percent()
        resource_text = f"{gpu_str} | CPU: {cpu_percent:.1f}%"
        cv2.putText(orig_frame, resource_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(orig_frame)
        
        # Batch log summary every 100 frames
        if frame_num % 200 == 0:
            logging.info(f"Processed {frame_num} frames, active tracks: {len(tracks)}")
    
    cap.release()
    out.release()
    logging.info(f"Processed video saved to {args.output_video_path}")
