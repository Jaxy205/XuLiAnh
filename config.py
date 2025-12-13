"""Tham số cấu hình cho Hệ thống Phát hiện Bất thường Lai"""

import torch


class Config:
    """Các tham số cấu hình hệ thống"""
    
    # Đầu vào Video
    VIDEO_PATH = "21.avi"  # Đổi thành 0 để dùng webcam
    OUTPUT_PATH = "output_anomaly_detection.mp4"
    
    # Cấu hình thiết bị (Sử dụng GPU nếu có)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cấu hình mô hình YOLOv8 (Nhận diện người)
    YOLO_MODEL = "yolov8n.pt"  # Model nano cho tốc độ nhanh
    YOLO_CONF_THRESHOLD = 0.5  # Ngưỡng tin cậy để coi là người
    PERSON_CLASS_ID = 0  # ID của lớp 'người' trong COCO dataset
    
    # Cấu hình theo dõi đối tượng (SORT Tracker)
    SORT_MAX_AGE = 30  # Số frame giữ ID khi không thấy đối tượng
    SORT_MIN_HITS = 3  # Số lần phát hiện tối thiểu để bắt đầu theo dõi
    SORT_IOU_THRESHOLD = 0.3 # Ngưỡng chồng lấn (IoU) để ghép cặp
    
    # ===== CÁC CHỨC NĂNG BẬT/TẮT =====
    ENABLE_RUNNING_DETECTION = True   # Phát hiện chạy
    ENABLE_FALLING_DETECTION = True   # Phát hiện ngã
    ENABLE_GATHERING_DETECTION = True # Phát hiện tụ tập
    ENABLE_TRAJECTORY_DETECTION = False # Tắt phát hiện quỹ đạo bất thường chung (chỉ dùng cho gathering)

    # Phát hiện chạy (Optical Flow - Luồng quang học)
    OPTICAL_FLOW_THRESHOLD = 3.0  # Ngưỡng độ lớn chuyển động trung bình
    
    # Phát hiện tụ tập (Khoảng cách Euclid)
    GATHERING_EPS = 50.0  # Khoảng cách tối đa (pixel) để coi là gần nhau
    GATHERING_MIN_SAMPLES = 3  # Số người tối thiểu để coi là tụ tập
    
    # Phát hiện ngã (Tỷ lệ khung hình - Aspect Ratio)
    # Nguyên lý: Khi ngã, chiều rộng > chiều cao (W/H > 1)
    FALL_RATIO_THRESHOLD = 1.2  # Tỷ lệ W/H để coi là ngã (lớn hơn 1 là nằm ngang)

    # Màu sắc khung bao (Định dạng BGR cho OpenCV)
    COLOR_FALLING = (0, 0, 255)    # Đỏ - Ưu tiên 1 (Ngã)
    COLOR_RUNNING = (0, 165, 255)  # Cam - Ưu tiên 2 (Chạy)
    COLOR_GATHERING = (0, 255, 255)  # Vàng - Ưu tiên 3 (Tụ tập)
    COLOR_TRAJECTORY_ANOMALY = (128, 0, 128) # Tím - Ưu tiên 4 (Quỹ đạo)
    COLOR_NORMAL = (0, 255, 0)     # Xanh lá - Bình thường
    
    # Hiển thị
    DISPLAY_FPS = True # Hiển thị tốc độ khung hình
    SAVE_OUTPUT = True # Lưu video kết quả
