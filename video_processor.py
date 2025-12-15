import torch.nn.functional as F
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
        
        res = yolo.track(frame, persist=True, classes=0, conf=args.conf_threshold, iou=args.iou_threshold)
        
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
            
            if len(tracks[tid]['embs']) >= args.min_embs_for_match:
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
                        if s > args.reid_threshold and s > mrs:
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
        
        ft_ = time.time() - fs
        ft.append(ft_)
        afps = 1 / (sum(ft) / len(ft)) if ft else 0
        cv2.putText(of, f"FPS: {afps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if 'cuda' in args.device:
            gu = torch.cuda.memory_allocated() / 1024**2
            gt = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gs = f"GPU: {gu:.0f}/{gt:.0f} MB"
        else:
            gs = "GPU: N/A"
        cp = psutil.cpu_percent()
        rt = f"{gs} | CPU: {cp:.1f}%"
        cv2.putText(of, rt, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(of)
        
        if fn % 100 == 0:
            logging.info(f"Proc {fn} f, act t: {len(tracks)}")
    
    cap.release()
    out.release()
    logging.info(f"Saved to {args.output_video_path}")