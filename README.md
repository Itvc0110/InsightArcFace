Cài đặt & Chạy thử
Cài đặt
Bashgit clone https://github.com/Itvc0110/InsightArcFace.git
cd InsightArcFace
pip install -r requirements.txt
Hướng dẫn cài checkpoint
InsightFace tự động tải model pack (e.g., buffalo_l) khi chạy lần đầu—không cần tải thủ công. # For future local checkpoint, put it in models/.
Hướng dẫn chạy thử
1. Trên ảnh tĩnh
Bashpython main.py --model_name buffalo_l \
    --input_dir ./input_images \
    --output_dir ./results/ \
    --batch_size 1024
2. Xây dựng Employee DB
Bashpython main.py --build_db \
    --model_name buffalo_l \
    --employees_dir ./employees \
    --db_path ./employee_db.json \
    --det_thresh 0.35
3. Chạy trên Video
Bashpython main.py --video_path ./input_video.mp4 \
    --model_name buffalo_l \
    --db_path ./employee_db.json \
    --output_video_path ./output_video.mp4 \
    --conf_threshold 0.5 \
    --cos_threshold 0.6 \
    --reid_threshold 0.8 \
    --skip_interval 5 \
    --det_thresh 0.35