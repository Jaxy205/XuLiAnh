import numpy as np
from .sort_tracker import Sort


class TrackerWrapper:
    """Lớp bao đóng bộ theo dõi SORT cải tiến (thay vì Centroid đơn giản)"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Khởi tạo bộ theo dõi
        
        Tham số:
            max_age: Số frame tối đa giữ ID khi không phát hiện được đối tượng
            min_hits: Số lần phát hiện tối thiểu để confirm ID
            iou_threshold: Ngưỡng IoU để ghép cặp
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Cập nhật bộ theo dõi với các phát hiện mới
        
        Tham số:
            detections: Mảng numpy shape (N, 5) -> [x1, y1, x2, y2, conf]
        
        Trả về:
            tracked_objects: Mảng numpy shape (N, 5) -> [x1, y1, x2, y2, id]
        """
        # SORT expects [x1, y1, x2, y2, score]
        
        if len(detections) == 0:
            tracks = self.tracker.update(np.empty((0, 5)))
        else:
            tracks = self.tracker.update(detections)
            
        # tracks trả về từ SORT là [x1, y1, x2, y2, id]
        # Định dạng này khớp với yêu cầu đầu ra của TrackerWrapper
        
        # Đảm bảo int cho tọa độ và id, giữ nguyên format array
        # Tuy nhiên SORT trả về float, nên ta có thể cast về int khi sử dụng hoặc ở đây
        # Output mong đợi là [x1, y1, x2, y2, id]
        
        return tracks

    def get_trajectories(self) -> dict:
        """
        Lấy dữ liệu quỹ đạo của các đối tượng đang theo dõi
        
        Trả về:
            trajectories: Dictionary {id: list_of_points}
        """
        return self.tracker.trajectories
