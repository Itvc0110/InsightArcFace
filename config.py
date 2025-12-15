import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='InsightFace Inference')
    parser.add_argument('--model_name', default='buffalo_l', type=str, help='InsightFace model pack (e.g., buffalo_l, buffalo_m)')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Not needed for InsightFace packs')
    parser.add_argument('--input_dir', type=str, default='./input_images', help='Directory containing input images for inference')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save embeddings')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--swap_color_channel', action='store_true', help='Swap color channels during preprocessing')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
    
    # Video pipeline args
    parser.add_argument('--video_path', type=str, default=None, help='Path to input video file')
    parser.add_argument('--db_path', type=str, default='./employee_db.json', help='Path to employee database JSON')
    parser.add_argument('--output_video_path', type=str, default='./output_video.mp4', help='Path to save annotated output video')
    parser.add_argument('--employees_dir', type=str, default=None, help='Directory for building employee DB')
    parser.add_argument('--build_db', action='store_true', help='Flag to build employee DB')

    # Hyperparams
    parser.add_argument('--conf_threshold', default=0.5, type=float, help='YOLO confidence threshold')
    parser.add_argument('--iou_threshold', default=0.45, type=float, help='YOLO IoU threshold')
    parser.add_argument('--upper_crop_ratio', default=0.4, type=float, help='Fraction of person bbox for head crop')
    parser.add_argument('--padding_ratio', default=0.2, type=float, help='Padding around crop')
    parser.add_argument('--skip_interval', default=5, type=int, help='Re-attempt embedding every N frames')
    parser.add_argument('--cos_threshold', default=0.6, type=float, help='Cosine similarity threshold for DB match')
    parser.add_argument('--min_embs_for_match', default=1, type=int, help='Min embeddings needed for matching')
    parser.add_argument('--min_crop_size', default=100, type=int, help='Min crop size in pixels')
    parser.add_argument('--max_embs_per_track', default=10, type=int, help='Max embeddings to average per track')
    parser.add_argument('--reid_threshold', default=0.8, type=float, help='Cosine similarity threshold for re-ID')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs including bboxes every frame')
    parser.add_argument('--log_level', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--max_missed_frames', default=3, type=int, help='Max frames to predict bbox before hiding')
    parser.add_argument('--det_thresh', default=0.35, type=float, help='Detection threshold for InsightFace')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args