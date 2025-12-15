import torch.nn.functional as F  
import time
import os
import numpy as np
import config
from data import get_inference_dataloader
from db_builder import build_employee_db
from video_processor import process_video

def main():
    args = config.get_args()
    
    if args.build_db:
        build_employee_db(args)
    elif args.video_path:
        process_video(args)
    else:
        print("Image inference not fully implemented for InsightFace; use video/DB modes.")

if __name__ == '__main__':
    main()