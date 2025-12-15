import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

# Global app instance
_app_instance = None

def _get_app(model_name='buffalo_l', det_thresh=0.35):
    global _app_instance
    if _app_instance is None:
        _app_instance = FaceAnalysis(name=model_name, providers=providers)
        _app_instance.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640), det_thresh=det_thresh)
        print("InsightFace buffalo_l model pack loaded once (GPU enabled)")
    return _app_instance

def get_embedding(image_path=None, bgr_np_image=None, model_name='buffalo_l', det_thresh=0.35):
    app = _get_app(model_name, det_thresh)
    
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = bgr_np_image
    
    if img is None:
        return None
    
    faces = app.get(img)
    if not faces:
        return None
    
    # Pick largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding.astype(np.float32)