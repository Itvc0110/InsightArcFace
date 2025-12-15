import os
import numpy as np
import json
from convert import get_embedding

def build_employee_db(args, model=None):  
    if not args.employees_dir or not os.path.isdir(args.employees_dir):
        raise ValueError(f"Employees directory not found: {args.employees_dir}")
    
    db = {}
    for name in os.listdir(args.employees_dir):
        dir_ = os.path.join(args.employees_dir, name)
        if not os.path.isdir(dir_): continue
        paths = [os.path.join(dir_, f) for f in os.listdir(dir_) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not paths:
            db[name] = []
            continue
        
        embs = []
        for p in paths:
            emb = get_embedding(image_path=p, model_name=args.model_name, det_thresh=args.det_thresh)
            if emb is not None:
                embs.append(emb)
        
        if embs:
            avg = np.mean(embs, axis=0).tolist()
            db[name] = avg
        
    with open(args.db_path, 'w') as f:
        json.dump(db, f)