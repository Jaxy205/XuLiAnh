"""Bộ phát hiện bất thường chính (Main Detector)"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple
import time

from ultralytics import YOLO

from .logic import (
    check_optical_flow, 
    check_gathering, 
    get_centroid,
    check_fall_simple,
)
from .tracker import TrackerWrapper
from .visualizer import Visualizer


class HybridAnomalyDetector:
    """Hệ thống chính tích hợp tất cả các module phát hiện"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Khởi tạo mô hình YOLOv8
        print("Đang tải mô hình YOLOv8...")
        self.yolo_model = YOLO(config.YOLO_MODEL)
        
        # Khởi tạo bộ theo dõi SORT
        print("Đang khởi tạo bộ theo dõi SORT...")
        self.tracker = TrackerWrapper(
            max_age=config.SORT_MAX_AGE,
            min_hits=config.SORT_MIN_HITS,
            iou_threshold=config.SORT_IOU_THRESHOLD
        )
        
        self.visualizer = Visualizer(config)
        
        
        # Frame trước đó để tính Optical Flow
        self.prev_gray = None
        
        self.process_count = 0
        
        print("✓ Hệ thống khởi tạo thành công!\n")
    
    def detect_fall(self, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """Phát hiện ngã đơn giản dựa trên tỷ lệ khung hình"""
        is_falling = check_fall_simple(bbox, threshold=self.config.FALL_RATIO_THRESHOLD)
        # Confidence giả lập là 1.0 nếu phát hiện ngã
        return is_falling, 1.0 if is_falling else 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Quy trình xử lý chính cho từng frame"""
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == self.config.PERSON_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                if conf >= self.config.YOLO_CONF_THRESHOLD:
                    detections.append([x1, y1, x2, y2, conf])
        
        detections = np.array(detections) if detections else np.empty((0, 5))
        
        tracked_objects = self.tracker.update(detections)
        
        # Cập nhật bộ đếm frame xử lý nội bộ
        self.process_count += 1
        

        centroid_list = []
        active_ids = []
        detection_results = {}
        
        for track in tracked_objects:
            x1, y1, x2, y2, person_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            person_id = int(person_id)
            active_ids.append(person_id)
            
            bbox = (x1, y1, x2, y2)
            centroid = get_centroid(bbox)
            centroid_list.append((person_id, centroid))
            
            status = "NORMAL"
            color = self.config.COLOR_NORMAL
            info = {}
            
            if self.config.ENABLE_FALLING_DETECTION:
                is_falling, fall_conf = self.detect_fall(bbox)
                if is_falling:
                    status = "FALLING"
                    color = self.config.COLOR_FALLING
                    info['fall_confidence'] = fall_conf
            
            if (status == "NORMAL" and self.config.ENABLE_RUNNING_DETECTION 
                and self.prev_gray is not None):
                is_running, flow_magnitude = check_optical_flow(
                    self.prev_gray, curr_gray, bbox,
                    threshold=self.config.OPTICAL_FLOW_THRESHOLD
                )
                
                # Kiểm tra thêm tỷ lệ khung hình: Người chạy phải ở tư thế đứng (W/H < 1.0)
                # Để tránh nhầm lẫn với người đang nằm/ngã nhưng có cử động
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                if is_running and aspect_ratio < 1.0:
                    status = "RUNNING"
                    color = self.config.COLOR_RUNNING
                    info['flow_magnitude'] = flow_magnitude

            
            detection_results[person_id] = (status, color, bbox, info)
        
        gathering_ids = set()
        gathering_groups = []
        if self.config.ENABLE_GATHERING_DETECTION:
            # Sử dụng quỹ đạo di chuyển (trajectories) để phát hiện tụ tập
            current_trajectories = self.tracker.get_trajectories()
            gathering_groups = check_gathering(
                current_trajectories,
                eps=self.config.GATHERING_EPS,
                min_samples=self.config.GATHERING_MIN_SAMPLES
            )
            
            for group in gathering_groups:
                gathering_ids.update(group)
            
            for person_id in gathering_ids:
                if person_id in detection_results:
                    status, color, bbox, info = detection_results[person_id]
                    if status not in ["FALLING", "RUNNING"]:
                        detection_results[person_id] = (
                            "GATHERING", 
                            self.config.COLOR_GATHERING, 
                            bbox, 
                            info
                        )
        
        annotated_frame = frame.copy()
        
        annotated_frame = self.visualizer.draw_individual_boxes(
            annotated_frame, detection_results, gathering_ids
        )
        annotated_frame = self.visualizer.draw_gathering_groups(
            annotated_frame, gathering_groups, detection_results
        )
        
        # Dọn dẹp
        self.prev_gray = curr_gray.copy()
        
        # Thống kê
        stats = {
            'total_persons': len(detection_results),
            'falling': sum(1 for s, _, _, _ in detection_results.values() if s == "FALLING"),
            'running': sum(1 for s, _, _, _ in detection_results.values() if s == "RUNNING"),
            'gathering': len(gathering_ids),
            'anomaly': sum(1 for s, _, _, _ in detection_results.values() if s == "ANOMALY"),
            'normal': sum(1 for s, _, _, _ in detection_results.values() if s == "NORMAL")
        }
        
        return annotated_frame, stats
    
    def run(self):
        """Vòng lặp thực thi chính"""
        
        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video {self.config.VIDEO_PATH}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        writer = None
        if self.config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                self.config.OUTPUT_PATH,
                fourcc,
                fps,
                (width, height)
            )
        
        frame_count = 0
        start_time = time.time()
        
        print("\n Bắt đầu xử lý video...\n")
        print("Các Module Phát hiện đang hoạt động:")
        print(f"   Phát hiện Chạy: {'BẬT' if self.config.ENABLE_RUNNING_DETECTION else 'TẮT'}")
        print(f"   Phát hiện Ngã: {'BẬT' if self.config.ENABLE_FALLING_DETECTION else 'TẮT'}")
        print(f"   Phát hiện Tụ tập: {'BẬT' if self.config.ENABLE_GATHERING_DETECTION else 'TẮT'}")

        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                annotated_frame, stats = self.process_frame(frame)
                
                if self.config.DISPLAY_FPS:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    annotated_frame = self.visualizer.draw_stats(
                        annotated_frame, frame_count, total_frames, current_fps, stats
                    )
                
                cv2.imshow('Hybrid Anomaly Detection', annotated_frame)
                
                if writer:
                    writer.write(annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠ Dừng bởi người dùng")
                    break
                
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Tiến độ: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        except KeyboardInterrupt:
            print("\n Bị ngắt bởi người dùng")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\n" + "="*80)
            print(" Xử lý hoàn tất!")
            print(f"  Tổng số frame: {frame_count}")
            print(f"  Thời gian: {elapsed_time:.2f}s")
            print(f"  FPS trung bình: {avg_fps:.2f}")
            if self.config.SAVE_OUTPUT:
                print(f"  Video kết quả đã lưu tại: {self.config.OUTPUT_PATH}")
            print("="*80)
