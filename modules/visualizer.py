"""Các chức năng hiển thị kết quả phát hiện"""

import cv2
import numpy as np
from typing import Dict, Tuple, List


class Visualizer:
    """Lớp xử lý việc vẽ và hiển thị các kết quả phát hiện lên hình ảnh"""
    
    def __init__(self, config):
        self.config = config
    
    def draw_individual_boxes(
        self, 
        frame: np.ndarray, 
        detection_results: Dict,
        gathering_ids: set
    ) -> np.ndarray:
        """
        Vẽ khung bao (bounding box) cho các cá nhân (không thuộc nhóm tụ tập).
        
        Args:
            frame: Ảnh đầu vào
            detection_results: Dict chứa kết quả phát hiện {id: (status, color, bbox, info)}
            gathering_ids: Set chứa ID của những người đang tụ tập (để bỏ qua không vẽ lẻ)
        """
        for person_id, (status, color, bbox, info) in detection_results.items():
            if person_id not in gathering_ids:
                # Chỉ vẽ nếu là RUNNING hoặc FALLING (theo yêu cầu người dùng)
                if status in ["RUNNING", "FALLING"]:
                    x1, y1, x2, y2 = bbox
                    
                    # Vẽ khung hình chữ nhật
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Vẽ nhãn thông tin
                    label = self._create_label(person_id, status, info)
                    self._draw_label(frame, label, x1, y1, color)
        
        return frame
            
    def draw_gathering_groups(
        self,
        frame: np.ndarray,
        gathering_groups: List[List[int]],
        detection_results: Dict
    ) -> np.ndarray:
        """
        Vẽ bao lồi (convex hull) bao quanh các nhóm tụ tập.
        """
        for group in gathering_groups:
            bboxes = [detection_results[pid][2] for pid in group if pid in detection_results]
            
            if bboxes:
                frame = self._draw_convex_hull(frame, bboxes, len(group))
        
        return frame
    
    def draw_stats(
        self,
        frame: np.ndarray,
        frame_count: int,
        total_frames: int,
        current_fps: float,
        stats: Dict
    ) -> np.ndarray:
        """
        Hiển thị FPS và các thông số thống kê lên góc màn hình.
        """
        # Nếu đã tắt FPS trong config (được check ở ngoài), hàm này có thể vẫn được gọi để vẽ stats khác
        # Tuy nhiên logic hiển thị FPS cụ thể nằm ở visualizer nên ta cứ vẽ nếu được truyền vào
        # Nhưng theo yêu cầu mới là tắt FPS, nên ta sẽ bỏ dòng FPS đi hoặc chỉ hiển thị Frame count
        
        # Cập nhật logic: Nếu current_fps = 0 hoặc config tắt, có thể ẩn. 
        # Nhưng ở đây ta cứ vẽ frame count.
        
        info_text = f"Frame: {frame_count}/{total_frames}"
        if current_fps > 0:
             info_text = f"FPS: {current_fps:.1f} | " + info_text
             
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        stats_text = (f"Persons: {stats['total_persons']} | "
                     f"Falling: {stats['falling']} | "
                     f"Running: {stats['running']} | "
                     f"Gathering: {stats['gathering']}")
        cv2.putText(
            frame, stats_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        return frame
    
    def _create_label(self, person_id: int, status: str, info: Dict) -> str:
        """Tạo nội dung nhãn text từ thông tin phát hiện"""
        label = f"ID:{person_id} {status}"
        if 'fall_confidence' in info:
            label += f" {info['fall_confidence']:.2f}"
        elif 'flow_magnitude' in info:
            label += f" {info['flow_magnitude']:.2f}"
        return label
    
    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: Tuple
    ) -> None:
        """Vẽ nhãn text có nền màu"""
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - 10),
            (x + label_size[0], y),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    def _draw_convex_hull(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple],
        group_size: int
    ) -> np.ndarray:
        """Vẽ đa giác bao lồi quanh nhóm tụ tập"""
        # Thu thập tất cả các điểm góc của các bbox trong nhóm
        all_points = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            all_points.extend([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        
        points_array = np.array(all_points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        
        # Vẽ đa giác đặc có độ trong suốt (transparency)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], color=self.config.COLOR_GATHERING)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        # Vẽ đường viền
        cv2.polylines(
            frame, [hull], isClosed=True,
            color=self.config.COLOR_GATHERING, thickness=3
        )
        
        # Vẽ nhãn nhóm ở tâm
        M = cv2.moments(hull)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            group_label = f"GATHERING (x{group_size})"
            label_size, _ = cv2.getTextSize(
                group_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (cx - label_size[0]//2 - 5, cy - label_size[1]//2 - 5),
                (cx + label_size[0]//2 + 5, cy + label_size[1]//2 + 5),
                self.config.COLOR_GATHERING, -1
            )
            cv2.putText(
                frame, group_label,
                (cx - label_size[0]//2, cy + label_size[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        
        return frame
