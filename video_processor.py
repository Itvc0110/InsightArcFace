import os
import numpy as np
import json
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
import psutil
import time
from convert import get_embedding
import warnings
from kalmanfilter import KalmanFilter
import torch

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Specified provider 'CUDAExecutionProvider' is not in available provider names")

# Setup logging
logging.basicConfig(level='INFO')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_video(args):
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if not args.video_path or not os.path.exists(args.video_path):
        raise ValueError(f"Video not found: {args.video_path}")
    
    with open(args.db_path, 'r') as f:
        db = json.load(f)
    db_embs = {n: np.array(e) for n, e in db.items() if e}
    if not db_embs:
        raise ValueError("Empty DB")
    
    yolo = YOLO('yolov8m.pt')
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError("Open video failed")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    known_p = {}
    next_pid = 0
    
    tracks = defaultdict(lambda: {'embs': deque(maxlen=args.max_embs_per_track), 'pid': None, 'fc': 0, 'kal': None, 'ld': None, 'mf': 0})
    
    fn = 0
    ft = deque(maxlen=10)
    
    while cap.isOpened():
        fs = time.time()
        ret, frame = cap.read()
        if not ret: break
        fn += 1
        of = frame.copy()
        
        # YOLO tracking time
        tracking_start = time.time()
        res = yolo.track(frame, persist=True, classes=0, conf=args.conf_threshold, iou=args.iou_threshold)
        tracking_time = time.time() - tracking_start
        
        # Recognition pipeline time (excluding vis)
        pipeline_start = time.time()
        dt = set()
        
        for r in res[0].boxes:
            if not r.id: continue
            tid = int(r.id)
            bb = r.xyxy[0].cpu().numpy()
            dt.add(tid)
            
            if tracks[tid]['kal'] is None:
                tracks[tid]['kal'] = KalmanFilter()
                meas = np.array([bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]], np.float32)
                tracks[tid]['kal'].kf.statePost = np.concatenate((meas, [0]*4)).astype(np.float32)
            
            meas = np.array([bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]], np.float32)
            tracks[tid]['kal'].update(meas)
            tracks[tid]['ld'] = bb
            tracks[tid]['mf'] = 0
            
            tracks[tid]['fc'] += 1
            
            if tracks[tid]['fc'] % args.skip_interval == 0:
                ph, pw = bb[3]-bb[1], bb[2]-bb[0]
                phd, pwd = ph * args.padding_ratio, pw * args.padding_ratio
                cy1 = max(0, int(bb[1] - phd))
                cy2 = min(h, int(bb[1] + ph * args.upper_crop_ratio + phd))
                cx1 = max(0, int(bb[0] - pwd))
                cx2 = min(w, int(bb[2] + pwd))
                
                if (cy2 - cy1 < args.min_crop_size) or (cx2 - cx1 < args.min_crop_size): continue
                
                cr = frame[cy1:cy2, cx1:cx2]  # BGR np
                
                emb = get_embedding(bgr_np_image=cr, model_name=args.model_name, det_thresh=args.det_thresh)
                if emb is None: continue
                
                tracks[tid]['embs'].append(emb)
            
            if len(tracks[tid]['embs']) >= args.min_embs_for_match or (tracks[tid]['fc'] % 20 == 0 and len(tracks[tid]['embs']) > 0):
                ae = np.mean(tracks[tid]['embs'], axis=0)
                
                mds = -1
                bn = None
                for n, de in db_embs.items():
                    s = cosine_similarity(ae, de)
                    if s > mds:
                        mds = s
                        bn = n
                nm = bn if mds > args.cos_threshold else None
                if nm:
                    logging.info(f"F {fn}, T {tid}: Match {nm} (s={mds:.2f})")
                
                mp = None
                mrs = -1
                for p, d in known_p.items():
                    if d['avg_emb'] is not None:
                        s = cosine_similarity(ae, d['avg_emb'])
                        if s > mrs:  # Removed reid_threshold; now highest sim wins if compatible
                            if (d['name'] == nm) or (d['name'] is None and nm is None):
                                mp = p
                                mrs = s
                                break
                            elif s > mrs:
                                mp = p
                                mrs = s
                
                if mp is not None:
                    tracks[tid]['pid'] = mp
                    oe = known_p[mp]['avg_emb']
                    known_p[mp]['avg_emb'] = (oe + ae) / 2
                    if nm and known_p[mp]['name'] is None:
                        known_p[mp]['name'] = nm
                    known_p[mp]['last_score'] = mds if nm else None
                    logging.info(f"F {fn}, T {tid}: Re-ID {mp} (s={mrs:.2f})")
                else:
                    pid = next_pid
                    next_pid += 1
                    known_p[pid] = {'avg_emb': ae, 'name': nm, 'last_score': mds if nm else None}
                    tracks[tid]['pid'] = pid
                    logging.info(f"F {fn}, T {tid}: New P {pid}")
        
        # Enforce max 1 ID per frame (max_id_per_frame = 1)
        pid_to_tracks = defaultdict(list)
        for tid in dt:
            if tracks[tid]['pid'] is not None:
                pid_to_tracks[tracks[tid]['pid']].append(tid)
        
        for pid, tlist in pid_to_tracks.items():
            if len(tlist) <= 1:
                continue  # No duplicate
                
            # Select primary: highest # embs → highest score → first
            primary_tid = max(
                tlist,
                key=lambda tid: (
                    len(tracks[tid]['embs']),  # Primary: most embeddings
                    known_p[pid]['last_score'] if known_p[pid]['last_score'] is not None else -1,  # Then highest score
                    tid  # Tiebreaker: lowest track ID (deterministic)
                )
            )
            
            primary_ae = np.mean(tracks[primary_tid]['embs'], axis=0) if tracks[primary_tid]['embs'] else None
            
            for dupe_tid in [t for t in tlist if t != primary_tid]:
                dupe_ae = np.mean(tracks[dupe_tid]['embs'], axis=0) if tracks[dupe_tid]['embs'] else None
                
                if primary_ae is not None and dupe_ae is not None:
                    sim = cosine_similarity(primary_ae, dupe_ae)
                    
                    if sim >= args.reid_threshold:
                        # Merge: append dupe embs to primary (deque handles maxlen)
                        for emb in tracks[dupe_tid]['embs']:
                            tracks[primary_tid]['embs'].append(emb)
                        
                        # Update name/score if dupe has better (or primary lacks)
                        if (known_p[pid]['name'] is None and 
                            tracks[dupe_tid].get('name') is not None):  # You'd need to store per-track name temporarily if needed
                            pass  # Or use known_p's current
                        if (known_p[pid]['last_score'] is None or 
                            tracks[dupe_tid].get('last_score', -1) > known_p[pid]['last_score']):
                            pass  # Update if better
                        
                        # Optional: update kalman with dupe's last detection (average or latest)
                        # For simplicity, keep primary's
                        
                        logging.info(f"F {fn}: Merge dupe T {dupe_tid} -> primary T {primary_tid} (sim={sim:.2f})")
                        
                        # Delete dupe track
                        del tracks[dupe_tid]
                    else:
                        # Split: assign new pid to dupe
                        new_pid = next_pid
                        next_pid += 1
                        known_p[new_pid] = {
                            'avg_emb': dupe_ae,
                            'name': known_p[pid]['name'],  # Inherit name (or None)
                            'last_score': known_p[pid]['last_score']
                        }
                        tracks[dupe_tid]['pid'] = new_pid
                        logging.info(f"F {fn}: Split dupe T {dupe_tid} -> New PID {new_pid} (sim={sim:.2f})")
                else:
                    # No reliable emb → keep primary, delete dupe
                    del tracks[dupe_tid]
                    logging.info(f"F {fn}: Delete dupe T {dupe_tid} (no emb)")

        for tid, s in list(tracks.items()):
            if tid not in dt:
                if s['kal'] and s['mf'] < args.max_missed_frames:
                    pr = s['kal'].predict()
                    px1, py1, pw, ph = pr[:4]
                    pb = np.array([px1, py1, px1 + pw, py1 + ph])
                    s['ld'] = pb
                    s['mf'] += 1
                else:
                    del tracks[tid]
                    continue
            
            bb = s['ld']
            if bb is not None:
                pid = s['pid']
                if pid is None:
                    lb = "No face detected"
                    sc = None
                    did = tid
                else:
                    pd = known_p[pid]
                    lb = pd['name'] if pd['name'] else "Unknown"
                    sc = pd['last_score']
                    did = pid
                
                if lb != "Unknown" and lb != "No face detected":
                    col = (0, 255, 0)
                elif lb == "Unknown":
                    col = (0, 255, 255)
                else:
                    col = (0, 0, 255)
                
                dlb = f"P {did}: {lb}"
                if sc is not None:
                    dlb += f" ({sc:.2f})"
                
                cv2.rectangle(of, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), col, 2)
                cv2.putText(of, dlb, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                
                if args.verbose:
                    logging.debug(f"F {fn}, T {tid}: BB {bb}")
        
        # FPS calculations (excluding vis)
        pipeline_time = time.time() - pipeline_start  # Recognition only
        whole_inference_time = tracking_time + pipeline_time  # YOLO + recognition
        tracking_fps = 1 / tracking_time if tracking_time > 0 else 0
        whole_fps = 1 / whole_inference_time if whole_inference_time > 0 else 0
        
        # Display FPS in red
        cv2.putText(of, f"Tracking FPS: {tracking_fps:.1f}", (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(of, f"Pipeline FPS: {whole_fps:.1f}", (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Resources (GPU/CPU/RAM) in white, bottom, 2 lines
        if 'cuda' in args.device:
            gu = torch.cuda.memory_allocated() / 1024**2  # Used in MB
            gt = torch.cuda.get_device_properties(0).total_memory / 1024**2  # Total in MB
            gpu_util = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0  # Compute %
            gpu_vram = f"{gu:.0f} / {gt / 1024:.0f} GB"
            gpu_usage = f" | Usage: {gpu_util}%"
        else:
            gpu_vram = "GPU: N/A"
            gpu_usage = " | Usage: N/A"
        
        cp = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        rt = f"{gpu_vram}{gpu_usage} | CPU: {cp:.1f}% | RAM: {ram:.1f}%"
        cv2.putText(of, rt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(of)
        
        if fn % 100 == 0:
            logging.info(f"Proc {fn} f, act t: {len(tracks)}")
    
    cap.release()
    out.release()
    logging.info(f"Saved to {args.output_video_path}")